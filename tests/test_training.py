"""Model-gradient and value-target tests for the truncated differentiable objective."""

import torch

from superm_repro.config import TrainConfig
from superm_repro.models import Critic, MotionPolicy, WorldModel
from superm_repro.replay import StartBatch, WindowBatch
from superm_repro.toy import PointMassTracking
from superm_repro.training import (
    fit_critic,
    fit_world_model,
    gae_targets,
    update_actor,
)


def test_gae_bootstraps_value_and_masks_terminal_transition() -> None:
    config = TrainConfig(gamma=0.9, gae_lambda=1.0)
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 10.0]])
    continuing = gae_targets(rewards, torch.zeros_like(rewards), values, config)
    torch.testing.assert_close(continuing, torch.tensor([[10.0, 10.0]]))

    terminal = gae_targets(rewards, torch.tensor([[0.0, 1.0]]), values, config)
    torch.testing.assert_close(terminal, torch.tensor([[1.9, 1.0]]))


def test_world_model_fits_autoregressive_windows() -> None:
    torch.manual_seed(2)
    task = PointMassTracking()
    model = WorldModel(task.spaces, hidden=(32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = torch.zeros(5, 6)
    context = torch.full((5, 1), 0.2)
    states = [state]
    actions = []
    for _ in range(4):
        action = torch.randn(5, 2)
        state = task.simulator_step(state, action, context)
        states.append(state)
        actions.append(action)
    batch = WindowBatch(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
        observations=torch.stack(states, dim=1),
        rewards=torch.zeros(5, 4),
        dones=torch.zeros(5, 4),
        contexts=context,
    )
    before = [p.detach().clone() for p in model.parameters()]
    loss = fit_world_model(model, batch, optimizer, TrainConfig())
    assert loss >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_bptt_updates_actor_but_not_frozen_model_or_critic() -> None:
    torch.manual_seed(3)
    task = PointMassTracking()
    model = WorldModel(task.spaces, hidden=(32,))
    actor = MotionPolicy(
        task.spaces, encoder_hidden=(32,), decoder_hidden=(32,)
    )
    critic = Critic(task.spaces, hidden=(32,))
    config = TrainConfig(imagination_horizon=4)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    starts = StartBatch(
        states=torch.zeros(4, 6),
        observation_history=torch.zeros(4, task.spaces.history, 6),
        references=torch.ones(4, config.imagination_horizon + task.spaces.goal_horizon, 2),
        contexts=torch.full((4, 1), 0.2),
    )

    before = [p.detach().clone() for p in actor.parameters()]
    rollout = update_actor(model, actor, critic, task, starts, optimizer, config)
    assert torch.isfinite(rollout.objective)
    assert any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in critic.parameters())

    loss = fit_critic(critic, rollout, critic_optimizer, config, epochs=1, minibatches=2)
    assert torch.isfinite(torch.tensor(loss))
