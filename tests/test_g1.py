"""Integration tests for the contact-rich Unitree G1 standing benchmark."""

import random

import pytest
import torch

from superm_repro.collect import collect_episode
from superm_repro.config import TrainConfig
from superm_repro.evaluation import evaluate_tracking_task
from superm_repro.models import Critic, MotionPolicy, WorldModel
from superm_repro.mujoco_g1 import MujocoG1Standing
from superm_repro.replay import ReplayBuffer
from superm_repro.train_g1 import NETWORK_SIZES
from superm_repro.training import fit_critic, fit_world_model, update_actor


def test_g1_loads_full_joint_model_and_step_executes_action() -> None:
    pytest.importorskip("mujoco")
    random.seed(2)
    task = MujocoG1Standing()
    assert (task.model.nq, task.model.nv, task.model.nu) == (36, 35, 29)
    assert task.model.npair > 0
    assert task.substeps == 5
    assert task.spaces.state == 100
    assert task.spaces.goal == 30

    goal = task.reference(20)
    initial = task.reset(goal[0], torch.ones(1))
    action = torch.linspace(-0.3, 0.3, 29)
    following = task.step(action)
    assert initial.shape == following.shape == (100,)
    torch.testing.assert_close(following[-29:], action)
    assert torch.isfinite(following).all()
    assert torch.isfinite(task.reward(following, action, goal[1]))
    assert task.data.time > 0.0


def test_g1_reward_and_termination_propagate_gradients() -> None:
    pytest.importorskip("mujoco")
    task = MujocoG1Standing()
    goal = task.reference(1)[0]
    state = task.reset(goal, torch.ones(1)).detach().requires_grad_(True)
    action = torch.ones(29, requires_grad=True) * 0.1
    value = task.reward(state, action, goal) - task.soft_done(state, goal)
    value.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    assert task.hard_done(state, goal).item() == 0.0
    fallen = state.detach().clone()
    fallen[2] = 0.35
    assert task.hard_done(fallen, goal).item() == 1.0


def test_g1_runs_replay_world_model_actor_and_critic() -> None:
    pytest.importorskip("mujoco")
    random.seed(10)
    torch.manual_seed(10)
    torch.set_num_threads(1)
    task = MujocoG1Standing()
    actor = MotionPolicy(task.spaces, encoder_hidden=(32,), decoder_hidden=(32,))
    model = WorldModel(task.spaces, hidden=(32,), quaternion_slice=(3, 7))
    critic = Critic(task.spaces, hidden=(32,))
    config = TrainConfig(imagination_horizon=4)
    replay = ReplayBuffer(capacity_steps=128)
    replay.add(collect_episode(task, actor, config, max_steps=64))
    assert replay.steps >= config.model_window

    loss = fit_world_model(
        model, replay.sample_windows(4, config.model_window),
        torch.optim.Adam(model.parameters(), lr=2e-4), config,
    )
    starts = replay.sample_starts(
        batch_size=4, horizon=config.imagination_horizon,
        history=task.spaces.history, goal_horizon=task.spaces.goal_horizon,
    )
    rollout = update_actor(
        model, actor, critic, task, starts,
        torch.optim.Adam(actor.parameters(), lr=2e-4), config,
    )
    critic_loss = fit_critic(
        critic, rollout, torch.optim.Adam(critic.parameters(), lr=2e-4), config,
    )
    assert torch.isfinite(torch.tensor([loss, critic_loss])).all()
    assert torch.isfinite(rollout.objective)
    assert rollout.states.shape == (4, 5, 100)
    torch.testing.assert_close(
        rollout.states[:, 1:, 3:7].norm(dim=-1), torch.ones(4, 4), atol=1e-5, rtol=0
    )


def test_g1_deterministic_evaluation_and_paper_network_widths() -> None:
    pytest.importorskip("mujoco")
    torch.manual_seed(14)
    task = MujocoG1Standing()
    actor = MotionPolicy(task.spaces, encoder_hidden=(32,), decoder_hidden=(32,))
    model = WorldModel(task.spaces, hidden=(32,))
    first = evaluate_tracking_task(task, actor, model, episodes=2, max_steps=64, seed=80)
    second = evaluate_tracking_task(task, actor, model, episodes=2, max_steps=64, seed=80)
    assert first == second
    assert first.mean_survival_steps > 0
    assert NETWORK_SIZES["paper"]["world"] == (2048, 2048, 2048)
    assert NETWORK_SIZES["paper"]["critic"] == (256, 128)
