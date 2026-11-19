"""Replay windows must remain contiguous and aligned to future reference time."""

import random

import torch

from superm_repro.replay import Episode, ReplayBuffer


def test_replay_window_and_reference_alignment() -> None:
    random.seed(7)
    timeline = torch.arange(13, dtype=torch.float32).unsqueeze(-1)
    episode = Episode(
        states=timeline,
        actions=timeline[:-1].clone(),
        observations=timeline.clone(),
        rewards=torch.arange(12, dtype=torch.float32),
        dones=torch.zeros(12),
        context=torch.tensor([0.8]),
        reference=torch.arange(80, dtype=torch.float32).unsqueeze(-1),
    )
    replay = ReplayBuffer(capacity_steps=24)
    replay.add(episode)
    windows = replay.sample_windows(batch_size=16, length=8)

    assert windows.states.shape == (16, 9, 1)
    torch.testing.assert_close(windows.states[:, :-1, 0], windows.actions[..., 0])
    torch.testing.assert_close(
        windows.states[:, 1:, 0] - windows.states[:, :-1, 0],
        torch.ones(16, 8),
    )
    starts = replay.sample_starts(batch_size=16, horizon=4, history=10, goal_horizon=10)
    torch.testing.assert_close(starts.states[:, 0], starts.references[:, 0, 0])
    torch.testing.assert_close(starts.states[:, 0], starts.observation_history[:, -1, 0])


def test_replay_removes_oldest_episode_without_splitting_windows() -> None:
    def episode(start: float) -> Episode:
        states = torch.arange(7, dtype=torch.float32).unsqueeze(-1) + start
        return Episode(
            states=states,
            actions=torch.zeros(6, 1),
            observations=states,
            rewards=torch.ones(6),
            dones=torch.zeros(6),
            context=torch.tensor([1.0]),
            reference=torch.zeros(60, 1),
        )

    replay = ReplayBuffer(capacity_steps=10)
    replay.add(episode(0.0))
    replay.add(episode(100.0))
    assert replay.steps == 6
    assert replay.sample_windows(5, 4).states.min().item() >= 100.0
