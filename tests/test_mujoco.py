"""Integration checks using actual MuJoCo rigid-body transitions."""

import pytest
import torch

from superm_repro.collect import collect_episode
from superm_repro.config import TrainConfig
from superm_repro.models import MotionPolicy
from superm_repro.mujoco_joint import MujocoJointTracking


def test_mujoco_collects_real_transitions_and_executed_actions() -> None:
    pytest.importorskip("mujoco")
    torch.manual_seed(1)
    task = MujocoJointTracking()
    actor = MotionPolicy(
        task.spaces, encoder_hidden=(16,), decoder_hidden=(16,)
    )
    config = TrainConfig(imagination_horizon=4)
    episode = collect_episode(task, actor, config, max_steps=24)

    assert episode.states.shape[0] == episode.length + 1
    assert episode.length >= 1
    assert episode.actions.shape[-1] == 2
    assert torch.isfinite(episode.states).all()
    assert torch.isfinite(episode.rewards).all()
    torch.testing.assert_close(episode.states[1:, 4:6], episode.actions)
    torch.testing.assert_close(episode.observations, episode.states)
    assert task.data.time > 0.0
