"""Integration checks using actual MuJoCo rigid-body transitions."""

import pytest
import torch

from superm_repro.collect import collect_episode
from superm_repro.config import TrainConfig
from superm_repro.evaluation import evaluate_cartpole
from superm_repro.models import MotionPolicy, WorldModel
from superm_repro.mujoco_cartpole import MujocoCartPole


def test_mujoco_collects_real_transitions_and_executed_actions() -> None:
    pytest.importorskip("mujoco")
    torch.manual_seed(1)
    task = MujocoCartPole()
    assert task.model.nq == 2
    assert task.model.nu == 1
    actor = MotionPolicy(
        task.spaces, encoder_hidden=(16,), decoder_hidden=(16,)
    )
    config = TrainConfig(imagination_horizon=4)
    episode = collect_episode(task, actor, config, max_steps=24)

    assert episode.states.shape[0] == episode.length + 1
    assert episode.length >= 1
    assert episode.actions.shape[-1] == 1
    assert torch.isfinite(episode.states).all()
    assert torch.isfinite(episode.rewards).all()
    torch.testing.assert_close(episode.states[1:, 4:5], episode.actions)
    torch.testing.assert_close(episode.observations, episode.states)
    assert task.data.time > 0.0


def test_cartpole_is_unstable_without_control() -> None:
    pytest.importorskip("mujoco")
    task = MujocoCartPole()
    goal = torch.zeros(2)
    task.reset(goal, torch.ones(1))
    task.data.qpos[1] = 0.1
    task.mujoco.mj_forward(task.model, task.data)
    initial_angle = abs(float(task.data.qpos[1]))
    for _ in range(40):
        task.step(torch.zeros(1))
    assert abs(float(task.data.qpos[1])) > initial_angle


def test_cartpole_deterministic_evaluation() -> None:
    pytest.importorskip("mujoco")
    torch.manual_seed(8)
    task = MujocoCartPole()
    actor = MotionPolicy(task.spaces, encoder_hidden=(16,), decoder_hidden=(16,))
    model = WorldModel(task.spaces, hidden=(16,))
    first = evaluate_cartpole(task, actor, model, episodes=2, max_steps=32, seed=19)
    second = evaluate_cartpole(task, actor, model, episodes=2, max_steps=32, seed=19)
    assert first == second
    assert 0.0 <= first.failure_rate <= 1.0
    assert first.mean_survival_steps > 0.0
    assert torch.isfinite(torch.tensor(first.model_state_mse))
