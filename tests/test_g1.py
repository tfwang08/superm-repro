"""Integration tests for the contact-rich Unitree G1 standing benchmark."""

import random

import numpy as np
import pytest
import torch

from superm_repro.collect import collect_episode
from superm_repro.config import TrainConfig
from superm_repro.evaluation import evaluate_tracking_task
from superm_repro.g1_kinematics import quat_angle
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
    assert task.spaces.observation == 93
    assert task.spaces.context == 33
    assert task.spaces.goal == 30

    goal = task.reference(20)
    context = task.sample_context()
    assert 0.3 <= context[0] <= 1.2
    assert context[1:30].abs().max() <= 0.01
    assert context[30:].abs().max() <= 0.05
    initial = task.reset(goal[0], context)
    np.testing.assert_allclose(
        task.model.pair_friction[task.foot_floor_pairs, 0], float(context[0])
    )
    np.testing.assert_allclose(
        task.model.body_ipos[task.torso_body_id],
        task.base_torso_ipos + context[30:33].numpy(),
    )
    np.testing.assert_allclose(
        task.data.ctrl,
        np.clip(
            task.stand_ctrl + context[1:30].numpy(),
            task.model.actuator_ctrlrange[:, 0],
            task.model.actuator_ctrlrange[:, 1],
        ),
    )
    action = torch.linspace(-0.3, 0.3, 29)
    following = task.step(action)
    assert initial.shape == following.shape == (100,)
    assert task.observation(following).shape == (93,)
    shifted = following.clone()
    shifted[:3] += torch.tensor([1.0, -2.0, 0.5])
    torch.testing.assert_close(task.observation(shifted), task.observation(following))
    torch.testing.assert_close(following[-29:], action)
    assert torch.isfinite(following).all()
    assert torch.isfinite(task.reward(following, action, goal[1]))
    assert task.data.time > 0.0


def test_g1_reward_and_termination_propagate_gradients() -> None:
    pytest.importorskip("mujoco")
    task = MujocoG1Standing()
    goal = task.reference(1)[0]
    state = task.reset(goal, task.evaluation_context()).detach().requires_grad_(True)
    action = torch.ones(29, requires_grad=True) * 0.1
    value = task.reward(state, action, goal, previous_state=state) - task.soft_done(state, goal)
    value.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    assert task.hard_done(state, goal).item() == 0.0
    fallen = state.detach().clone()
    fallen[2] = float(goal[0]) - 0.26
    assert task.hard_done(fallen, goal).item() == 1.0
    leaning = state.detach().clone()
    leaning[3:7] = torch.tensor([np.cos(0.75), np.sin(0.75), 0.0, 0.0])
    assert task.hard_done(leaning, goal).item() == 1.0

    home = torch.as_tensor(
        np.concatenate((task.stand_pose, np.zeros(35), np.zeros(29))), dtype=torch.float32
    )
    home_reward = task.reward(home, torch.zeros(29), goal, previous_state=home)
    torch.testing.assert_close(home_reward, torch.tensor(0.1), atol=1e-5, rtol=0)
    previous_action = home.clone()
    previous_action[-29:] = 1.0
    assert task.reward(home, torch.zeros(29), goal, previous_state=previous_action) < home_reward


def test_g1_observation_noise_matches_paper_ranges() -> None:
    pytest.importorskip("mujoco")
    task = MujocoG1Standing()
    ideal = task.observation(task.reset(task.reference(1)[0], task.evaluation_context()))
    noise = task.observation_noise(ideal.repeat(100, 1))
    assert noise[:, :29].abs().max() <= 0.01
    assert noise[:, 29:58].abs().max() <= 0.5
    assert noise[:, 58:61].abs().max() <= 0.2
    assert noise[:, 61:64].abs().max() <= 0.05
    assert noise[:, 64:].abs().max() == 0
    torch.testing.assert_close(
        task.observation(task._state())[61:64], torch.tensor([0.0, 0.0, -1.0]), atol=0.1, rtol=0
    )


def test_g1_differentiable_fk_matches_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    task = MujocoG1Standing()
    task.reset(task.reference(1)[0], task.evaluation_context())
    rng = np.random.default_rng(42)
    task.data.qpos[7:] += rng.uniform(-0.13, 0.13, 29)
    task.data.qvel[:] = rng.uniform(-0.15, 0.15, 35)
    mujoco.mj_forward(task.model, task.data)
    forward = task.kinematics.forward(task._state())
    np.testing.assert_allclose(forward.positions.numpy(), task.data.xpos[1:], atol=2e-6)
    assert (
        quat_angle(
            forward.quaternions, torch.tensor(task.data.xquat[1:].copy(), dtype=torch.float32)
        ).max()
        < 1e-5
    )

    reference_positions = task.data.xpos[1:].copy()
    integrated = task.data.qpos.copy()
    mujoco.mj_integratePos(task.model, integrated, task.data.qvel, 1e-6)
    future = mujoco.MjData(task.model)
    future.qpos[:] = integrated
    mujoco.mj_forward(task.model, future)
    finite_difference = (future.xpos[1:] - reference_positions) / 1e-6
    np.testing.assert_allclose(forward.linear_velocities.numpy(), finite_difference, atol=1e-5)


def test_g1_runs_replay_world_model_actor_and_critic() -> None:
    pytest.importorskip("mujoco")
    random.seed(10)
    torch.manual_seed(10)
    torch.set_num_threads(1)
    task = MujocoG1Standing()
    actor = MotionPolicy(task.spaces, encoder_hidden=(32,), decoder_hidden=(32,))
    model = WorldModel(
        task.spaces,
        hidden=(32,),
        quaternion_slice=(3, 7),
        action_state_slice=(71, 100),
    )
    critic = Critic(task.spaces, hidden=(32,))
    config = TrainConfig(imagination_horizon=4)
    replay = ReplayBuffer(capacity_steps=128)
    replay.add(collect_episode(task, actor, config, max_steps=64))
    assert replay.steps >= config.model_window
    with torch.no_grad():
        model_step = model(
            replay.episodes[0].states[:1],
            replay.episodes[0].actions[:1],
            replay.episodes[0].context[None, :],
        )
    torch.testing.assert_close(model_step[..., -29:], replay.episodes[0].actions[:1])

    loss = fit_world_model(
        model,
        replay.sample_windows(4, config.model_window),
        torch.optim.Adam(model.parameters(), lr=2e-4),
        config,
    )
    starts = replay.sample_starts(
        batch_size=4,
        horizon=config.imagination_horizon,
        history=task.spaces.history,
        goal_horizon=task.spaces.goal_horizon,
    )
    rollout = update_actor(
        model,
        actor,
        critic,
        task,
        starts,
        torch.optim.Adam(actor.parameters(), lr=2e-4),
        config,
    )
    critic_loss = fit_critic(
        critic,
        rollout,
        torch.optim.Adam(critic.parameters(), lr=2e-4),
        config,
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
