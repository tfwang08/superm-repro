"""Paper-aligned k=8 model fitting and H=32 differentiable actor-critic updates."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol

import torch
from torch import Tensor, nn

from .config import TrainConfig
from .models import Critic, MotionPolicy, WorldModel
from .replay import StartBatch, WindowBatch


class DifferentiableTask(Protocol):
    def observation(self, state: Tensor) -> Tensor: ...

    def observation_noise(self, ideal: Tensor) -> Tensor: ...

    def reward(self, next_state: Tensor, action: Tensor, next_goal: Tensor) -> Tensor: ...

    def soft_done(self, next_state: Tensor, next_goal: Tensor) -> Tensor: ...


@dataclass
class ImaginedRollout:
    states: Tensor  # [B, H+1, state_dim]
    rewards: Tensor  # [B, H]
    dones: Tensor  # [B, H], differentiable termination probabilities
    references: Tensor  # [B, H+K, goal_dim]
    contexts: Tensor  # [B, context_dim]
    objective: Tensor  # Scalar, including terminal critic value


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    """Freeze weights while retaining derivatives through the module's inputs."""
    parameters = list(module.parameters())
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in zip(parameters, previous):
            parameter.requires_grad_(required)


def world_model_loss(model: WorldModel, batch: WindowBatch) -> Tensor:
    """Autoregressive k-step state loss with only the first state from simulation.

    The first milestone computes observation, reward, and termination analytically.
    Therefore their supervised losses are intentionally omitted (paper Eq. 1).
    """
    prediction = batch.states[:, 0]
    losses = []
    for step in range(batch.actions.shape[1]):
        prediction = model(prediction, batch.actions[:, step], batch.contexts)
        losses.append((prediction - batch.states[:, step + 1]).square().sum(dim=-1))
    return torch.stack(losses, dim=1).sum(dim=1).mean()


def fit_world_model(
    model: WorldModel,
    batch: WindowBatch,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = world_model_loss(model, batch)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.model_grad_clip)
    optimizer.step()
    return loss.item()


def imagine(
    model: WorldModel,
    actor: MotionPolicy,
    critic: Critic,
    task: DifferentiableTask,
    starts: StartBatch,
    config: TrainConfig,
) -> ImaginedRollout:
    """Compute the differentiable truncated return plus V(s_H, g_H, xi)."""
    horizon = config.imagination_horizon
    goal_horizon = actor.spaces.goal_horizon
    if starts.references.shape[1] < horizon + goal_horizon:
        raise ValueError("Not enough future reference frames for imagination")

    state = starts.states
    history = starts.observation_history
    history = history + task.observation_noise(history)
    survival = torch.ones_like(state[:, 0])
    discounted_return = torch.zeros_like(survival)
    states = [state]
    rewards = []
    dones = []

    for t in range(horizon):
        future_goals = starts.references[:, t : t + goal_horizon]
        action = actor(history, future_goals)
        next_state = model(state, action, starts.contexts)
        next_goal = starts.references[:, t + 1]
        reward = task.reward(next_state, action, next_goal)
        done = task.soft_done(next_state, next_goal)

        discounted_return = discounted_return + (config.gamma**t) * survival * reward
        survival = survival * (1.0 - done)
        states.append(next_state)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        if t + 1 < horizon:
            observation = task.observation(state)
            noisy_observation = observation + task.observation_noise(observation)
            history = torch.cat((history[:, 1:], noisy_observation.unsqueeze(1)), dim=1)

    terminal_goals = starts.references[:, horizon : horizon + goal_horizon]
    bootstrap = critic(state, terminal_goals, starts.contexts)
    discounted_return = discounted_return + (config.gamma**horizon) * survival * bootstrap
    return ImaginedRollout(
        states=torch.stack(states, dim=1),
        rewards=torch.stack(rewards, dim=1),
        dones=torch.stack(dones, dim=1),
        references=starts.references,
        contexts=starts.contexts,
        objective=discounted_return.mean(),
    )


@torch.no_grad()
def gae_targets(rewards: Tensor, dones: Tensor, values: Tensor, config: TrainConfig) -> Tensor:
    """Compute Eq. 3 using only imagined rewards and a bootstrapped final value."""
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("Expected one additional terminal critic value")
    if dones.shape != rewards.shape:
        raise ValueError("Termination probabilities must match rewards")
    next_return = values[:, -1]
    targets = []
    for t in reversed(range(rewards.shape[1])):
        next_return = rewards[:, t] + config.gamma * (1.0 - dones[:, t]) * (
            (1.0 - config.gae_lambda) * values[:, t + 1]
            + config.gae_lambda * next_return
        )
        targets.append(next_return)
    return torch.stack(list(reversed(targets)), dim=1)


def update_actor(
    model: WorldModel,
    actor: MotionPolicy,
    critic: Critic,
    task: DifferentiableTask,
    starts: StartBatch,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> ImaginedRollout:
    optimizer.zero_grad(set_to_none=True)
    # The world-model and critic weights remain fixed in the actor's backward pass.
    # Do not use torch.no_grad(): gradients must traverse both networks' inputs.
    with frozen(model), frozen(critic):
        rollout = imagine(model, actor, critic, task, starts, config)
        (-rollout.objective).backward()
    nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
    for parameter in actor.parameters():
        if parameter.grad is not None:
            parameter.grad.clamp_(-0.1, 0.1)
    optimizer.step()
    return rollout


def fit_critic(
    critic: Critic,
    rollout: ImaginedRollout,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epochs: int = 5,
    minibatches: int = 4,
) -> float:
    """Fit one critic to fixed GAE targets from a previously imagined trajectory."""
    states = rollout.states.detach()
    rewards = rollout.rewards.detach()
    dones = rollout.dones.detach()
    horizon = rewards.shape[1]
    batch_size = rewards.shape[0]
    goal_horizon = critic.spaces.goal_horizon
    future_goals = torch.stack(
        [rollout.references[:, t : t + goal_horizon] for t in range(horizon + 1)],
        dim=1,
    ).detach()
    contexts = rollout.contexts.detach().unsqueeze(1).expand(-1, horizon + 1, -1)

    with torch.no_grad():
        initial_values = critic(
            states.reshape(-1, states.shape[-1]),
            future_goals.reshape(-1, goal_horizon, future_goals.shape[-1]),
            contexts.reshape(-1, contexts.shape[-1]),
        ).reshape(batch_size, horizon + 1)
        targets = gae_targets(rewards, dones, initial_values, config)

    total_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for indices in torch.tensor_split(torch.randperm(batch_size, device=states.device), minibatches):
            if indices.numel() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            selected_states = states[indices, :-1]
            selected_goals = future_goals[indices, :-1]
            selected_contexts = contexts[indices, :-1]
            values = critic(
                selected_states.reshape(-1, states.shape[-1]),
                selected_goals.reshape(-1, goal_horizon, future_goals.shape[-1]),
                selected_contexts.reshape(-1, contexts.shape[-1]),
            ).reshape(indices.numel(), horizon)
            loss = (values - targets[indices]).square().sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), config.critic_grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)
