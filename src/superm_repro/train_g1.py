"""Train and validate a SuperMimic-style G1 standing controller in MuJoCo."""

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .collect import collect_episode
from .config import TrainConfig
from .evaluation import evaluate_tracking_task
from .models import Critic, MotionPolicy, WorldModel
from .mujoco_g1 import MujocoG1Standing
from .replay import ReplayBuffer
from .training import fit_critic, fit_world_model, update_actor

NETWORK_SIZES = {
    "tiny": {
        "world": (64, 64),
        "encoder": (64, 32),
        "decoder": (64, 32),
        "critic": (64, 32),
    },
    "small": {
        "world": (256, 256),
        "encoder": (256, 128),
        "decoder": (256, 128),
        "critic": (256, 128),
    },
    "paper": {
        "world": (2048, 2048, 2048),
        "encoder": (2048, 1024, 512, 256),
        "decoder": (2048, 1024, 512, 256),
        "critic": (256, 128),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--initial-episodes", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--pretrain-model-updates", type=int, default=100)
    parser.add_argument("--model-updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--network-size", choices=NETWORK_SIZES, default="small")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--long-test", action="store_true")
    parser.add_argument("--long-test-episodes", type=int, default=32)
    parser.add_argument("--long-test-steps", type=int, default=500)
    parser.add_argument("--test-seed", type=int, default=2601)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/g1_standing.pt")
    args = parser.parse_args()

    if (
        min(
            args.iterations,
            args.initial_episodes,
            args.episode_steps,
            args.model_updates,
            args.batch_size,
            args.horizon,
            args.checkpoint_every,
            args.eval_episodes,
            args.eval_steps,
            args.long_test_episodes,
            args.long_test_steps,
            args.torch_threads,
        )
        < 1
        or args.pretrain_model_updates < 0
    ):
        parser.error("Training and evaluation sizes must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    task = MujocoG1Standing(scene_path=args.scene)
    validation_task = MujocoG1Standing(scene_path=args.scene)
    sizes = NETWORK_SIZES[args.network_size]
    config = replace(TrainConfig(), imagination_horizon=args.horizon)
    model = WorldModel(
        task.spaces,
        hidden=sizes["world"],
        quaternion_slice=(3, 7),
        action_state_slice=(71, 100),
    ).to(device)
    actor = MotionPolicy(
        task.spaces,
        encoder_hidden=sizes["encoder"],
        decoder_hidden=sizes["decoder"],
    ).to(device)
    critic = Critic(task.spaces, hidden=sizes["critic"]).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity_steps=max(65_536, args.episode_steps * 2))

    initial_evaluation = evaluate_tracking_task(
        validation_task,
        actor,
        model,
        episodes=args.eval_episodes,
        max_steps=args.eval_steps,
        seed=args.validation_seed,
    )
    print(f"Initial deterministic G1 evaluation: {initial_evaluation}", flush=True)

    for _ in range(args.initial_episodes):
        replay.add(collect_episode(task, actor, config, args.episode_steps))
    print(f"Initial MuJoCo replay: {replay.steps} transitions; device={device}", flush=True)

    for _ in range(args.pretrain_model_updates):
        windows = replay.sample_windows(args.batch_size, config.model_window).to(device)
        fit_world_model(model, windows, model_optimizer, config)

    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_checkpoint: Path | None = None
    validation_history = []

    for iteration in range(1, args.iterations + 1):
        episode = collect_episode(task, actor, config, args.episode_steps)
        replay.add(episode)

        losses = []
        for _ in range(args.model_updates):
            windows = replay.sample_windows(args.batch_size, config.model_window).to(device)
            losses.append(fit_world_model(model, windows, model_optimizer, config))

        starts = replay.sample_starts(
            args.batch_size,
            config.imagination_horizon,
            task.spaces.history,
            task.spaces.goal_horizon,
        ).to(device)
        rollout = update_actor(model, actor, critic, task, starts, actor_optimizer, config)
        critic_loss = fit_critic(critic, rollout, critic_optimizer, config)
        print(
            f"iter={iteration:04d} replay={replay.steps:6d} "
            f"sim_steps={episode.length:3d} "
            f"sim_reward={episode.rewards.mean().item():.3f} "
            f"wm_loss={np.mean(losses):.4f} "
            f"imagined_return={rollout.objective.item():.3f} "
            f"critic_loss={critic_loss:.3f}",
            flush=True,
        )

        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            candidate = output.with_name(f"{output.stem}_iter{iteration:04d}{output.suffix}")
            torch.save(
                {
                    "world_model": model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "config": asdict(config),
                    "task": "g1_standing",
                    "network_size": args.network_size,
                    "scene": str(args.scene) if args.scene is not None else "bundled_physics_only",
                    "links": 29,
                    "seed": args.seed,
                    "iterations": iteration,
                },
                candidate,
            )
            report = evaluate_tracking_task(
                validation_task,
                actor,
                model,
                episodes=args.eval_episodes,
                max_steps=args.eval_steps,
                model_window=config.model_window,
                seed=args.validation_seed,
            )
            print(f"Independent validation at iteration {iteration}: {report}", flush=True)
            validation_history.append(
                {"iteration": iteration, "checkpoint": str(candidate), **asdict(report)}
            )
            if report.mean_normalized_return > best_score:
                best_score = report.mean_normalized_return
                best_checkpoint = candidate

    if best_checkpoint is None:
        raise RuntimeError("No G1 validation checkpoint was produced")
    selected = torch.load(best_checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(selected["actor"])
    model.load_state_dict(selected["world_model"])
    critic.load_state_dict(selected["critic"])
    actor.eval()
    model.eval()
    critic.eval()
    torch.save(selected, output)
    print(f"Selected independent-validation checkpoint: {best_checkpoint}", flush=True)

    summary = {
        "task": "g1_standing",
        "seed": args.seed,
        "network_size": args.network_size,
        "scene": str(args.scene) if args.scene is not None else "bundled_physics_only",
        "horizon": args.horizon,
        "control_dt": task.control_dt,
        "initial_validation": asdict(initial_evaluation),
        "selected_iteration": selected["iterations"],
        "validation_history": validation_history,
    }
    if args.long_test:
        held_out_task = MujocoG1Standing(scene_path=args.scene)
        score = evaluate_tracking_task(
            held_out_task,
            actor,
            model,
            episodes=args.long_test_episodes,
            max_steps=args.long_test_steps,
            model_window=config.model_window,
            seed=args.test_seed,
        )
        summary["long_test"] = asdict(score)
        survivors = round(args.long_test_episodes * (1.0 - score.failure_rate))
        print(f"Frozen G1 survival: {survivors}/{args.long_test_episodes}", flush=True)
        print(f"Frozen G1 long-test metrics: {score}", flush=True)

    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved selected checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
