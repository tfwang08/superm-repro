"""Run the SuperMimic-style training loop on a MuJoCo inverted pendulum."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .collect import collect_episode
from .config import TrainConfig
from .evaluation import evaluate_cartpole
from .models import Critic, MotionPolicy, WorldModel
from .mujoco_cartpole import MujocoCartPole
from .replay import ReplayBuffer
from .training import fit_critic, fit_world_model, update_actor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=320)
    parser.add_argument("--initial-episodes", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--model-updates", type=int, default=10)
    parser.add_argument("--pretrain-model-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--profile", choices=("baseline", "experiment"), default="experiment")
    parser.add_argument("--links", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--relative-goal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--antisymmetric", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--motion", choices=("balance", "track", "step", "reverse"), default=None
    )
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--long-test", action="store_true")
    parser.add_argument("--long-test-episodes", type=int, default=96)
    parser.add_argument("--long-test-steps", type=int, default=512)
    parser.add_argument("--test-seed", type=int, default=2601)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/mujoco_cartpole.pt")
    args = parser.parse_args()
    if min(
        args.iterations,
        args.initial_episodes,
        args.episode_steps,
        args.model_updates,
        args.batch_size,
        args.eval_episodes,
        args.eval_steps,
        args.checkpoint_every,
        args.torch_threads,
        args.long_test_episodes,
        args.long_test_steps,
    ) < 1:
        parser.error("All training sizes must be positive")
    if args.pretrain_model_updates < 0:
        parser.error("Pretraining update count cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    train_motion = args.motion or ("track" if args.profile == "baseline" else "step")
    relative_goal = args.relative_goal if args.relative_goal is not None else args.profile == "experiment"
    antisymmetric = (
        args.antisymmetric if args.antisymmetric is not None else args.profile == "experiment"
    )
    task = MujocoCartPole(motion=train_motion, links=args.links)
    validation_task = MujocoCartPole(motion="step", links=args.links)
    spaces = task.spaces
    config = TrainConfig()

    # Small networks make this stage practical on a laptop CPU. The model classes
    # themselves default to the much larger widths reported in Appendix C.
    world_model = WorldModel(spaces, hidden=(128, 128)).to(device)
    actor = MotionPolicy(
        spaces, encoder_hidden=(128, 64), decoder_hidden=(128, 64),
        relative_goal=relative_goal,
        antisymmetric=antisymmetric,
    ).to(device)
    critic = Critic(spaces, hidden=(64, 64)).to(device)
    model_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.learning_rate)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity_steps=max(16_384, args.episode_steps * 2))

    initial_evaluation = evaluate_cartpole(
        validation_task, actor, world_model,
        episodes=args.eval_episodes, max_steps=args.eval_steps,
        seed=args.validation_seed,
    )
    print(f"Initial deterministic evaluation: {initial_evaluation}")

    for _ in range(args.initial_episodes):
        replay.add(collect_episode(task, actor, config, args.episode_steps))
    print(f"Initial simulator replay: {replay.steps} transitions; device={device}")

    # A warm start reduces exploitation of a nearly random dynamics surrogate.
    for _ in range(args.pretrain_model_updates):
        windows = replay.sample_windows(args.batch_size, config.model_window).to(device)
        fit_world_model(world_model, windows, model_optimizer, config)

    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_checkpoint: Path | None = None
    validation_history = []

    for iteration in range(1, args.iterations + 1):
        # Phase 1: collect fresh on-policy simulator transitions with action noise.
        episode = collect_episode(task, actor, config, args.episode_steps)
        replay.add(episode)

        # Phase 2: fit dynamics on contiguous, logged, autoregressive k=8 windows.
        model_losses = []
        for _ in range(args.model_updates):
            windows = replay.sample_windows(args.batch_size, config.model_window).to(device)
            model_losses.append(fit_world_model(world_model, windows, model_optimizer, config))

        # Phase 3: maximize 32-step imagined return and fit the bootstrap critic.
        starts = replay.sample_starts(
            args.batch_size,
            config.imagination_horizon,
            spaces.history,
            spaces.goal_horizon,
        ).to(device)
        rollout = update_actor(
            world_model, actor, critic, task, starts, actor_optimizer, config
        )
        critic_loss = fit_critic(critic, rollout, critic_optimizer, config)

        realized_reward = episode.rewards.mean().item()
        print(
            f"iter={iteration:04d} replay={replay.steps:5d} "
            f"sim_reward={realized_reward:.3f} "
            f"wm_loss={np.mean(model_losses):.4f} "
            f"imagined_return={rollout.objective.item():.3f} "
            f"critic_loss={critic_loss:.3f}"
        )

        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            candidate = output.with_name(
                f"{output.stem}_iter{iteration:04d}{output.suffix}"
            )
            torch.save(
                {
                    "world_model": world_model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "config": vars(config),
                    "profile": args.profile,
                    "links": args.links,
                    "motion": train_motion,
                    "relative_goal": relative_goal,
                    "antisymmetric": antisymmetric,
                    "seed": args.seed,
                    "iterations": iteration,
                },
                candidate,
            )
            report = evaluate_cartpole(
                validation_task,
                actor,
                world_model,
                episodes=args.eval_episodes,
                max_steps=args.eval_steps,
                model_window=config.model_window,
                seed=args.validation_seed,
            )
            print(f"Independent validation at iteration {iteration}: {report}")
            validation_history.append(
                {"iteration": iteration, "checkpoint": str(candidate), **asdict(report)}
            )
            if report.mean_normalized_return > best_score:
                best_score = report.mean_normalized_return
                best_checkpoint = candidate

    if best_checkpoint is None:
        raise RuntimeError("No validation checkpoint was produced")
    selected = torch.load(best_checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(selected["actor"])
    world_model.load_state_dict(selected["world_model"])
    critic.load_state_dict(selected["critic"])
    actor.eval()
    world_model.eval()
    critic.eval()
    torch.save(selected, output)
    print(f"Selected checkpoint using independent validation: {best_checkpoint}")

    summary = {
        "profile": args.profile,
        "links": args.links,
        "seed": args.seed,
        "training_motion": train_motion,
        "relative_goal": relative_goal,
        "antisymmetric": antisymmetric,
        "initial_validation": asdict(initial_evaluation),
        "selected_iteration": selected["iterations"],
        "validation_history": validation_history,
    }
    if args.long_test:
        held_out_task = MujocoCartPole(motion="reverse", links=args.links)
        held_out = evaluate_cartpole(
            held_out_task,
            actor,
            world_model,
            episodes=args.long_test_episodes,
            max_steps=args.long_test_steps,
            model_window=config.model_window,
            seed=args.test_seed,
        )
        summary["long_test"] = asdict(held_out)
        successes = round(args.long_test_episodes * (1.0 - held_out.failure_rate))
        print(f"Frozen checkpoint repeated-reversal success: {successes}/{args.long_test_episodes}")
        print(f"Frozen long-test metrics: {held_out}")

    report_file = output.with_suffix(".json")
    report_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved selected checkpoint: {output}")
    print(f"Saved experiment report: {report_file}")


if __name__ == "__main__":
    main()
