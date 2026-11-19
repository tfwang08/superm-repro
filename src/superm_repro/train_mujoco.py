"""Run the SuperMimic-style training loop on a MuJoCo inverted pendulum."""

import argparse
import random
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
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--initial-episodes", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--model-updates", type=int, default=10)
    parser.add_argument("--pretrain-model-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--motion", choices=("balance", "track"), default="balance")
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
        args.eval_every,
        args.torch_threads,
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
    task = MujocoCartPole(motion=args.motion)
    spaces = task.spaces
    config = TrainConfig()

    # Small networks make this stage practical on a laptop CPU. The model classes
    # themselves default to the much larger widths reported in Appendix C.
    world_model = WorldModel(spaces, hidden=(128, 128)).to(device)
    actor = MotionPolicy(
        spaces, encoder_hidden=(128, 64), decoder_hidden=(128, 64)
    ).to(device)
    critic = Critic(spaces, hidden=(64, 64)).to(device)
    model_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.learning_rate)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity_steps=max(16_384, args.episode_steps * 2))

    initial_evaluation = evaluate_cartpole(
        task, actor, world_model, episodes=args.eval_episodes, max_steps=args.eval_steps
    )
    print(f"Initial deterministic evaluation: {initial_evaluation}")

    for _ in range(args.initial_episodes):
        replay.add(collect_episode(task, actor, config, args.episode_steps))
    print(f"Initial simulator replay: {replay.steps} transitions; device={device}")

    # A warm start reduces exploitation of a nearly random dynamics surrogate.
    for _ in range(args.pretrain_model_updates):
        windows = replay.sample_windows(args.batch_size, config.model_window).to(device)
        fit_world_model(world_model, windows, model_optimizer, config)

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

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            report = evaluate_cartpole(
                task,
                actor,
                world_model,
                episodes=args.eval_episodes,
                max_steps=args.eval_steps,
                model_window=config.model_window,
            )
            print(f"Deterministic evaluation at iteration {iteration}: {report}")

    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": vars(config),
            "seed": args.seed,
            "iterations": args.iterations,
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
