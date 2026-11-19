"""Gather policy-generated, action-noised transitions from an external simulator."""

from typing import Protocol

import torch
from torch import Tensor

from .config import TrainConfig
from .models import MotionPolicy
from .replay import Episode


class TrackingSimulator(Protocol):
    spaces: object

    def reference(self, length: int) -> Tensor: ...

    def sample_context(self) -> Tensor: ...

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor: ...

    def step(self, executed_action: Tensor) -> Tensor: ...

    def observation(self, state: Tensor) -> Tensor: ...

    def observation_noise(self, ideal: Tensor) -> Tensor: ...

    def reward(
        self, next_state: Tensor, action: Tensor, next_goal: Tensor,
        previous_state: Tensor | None = None,
    ) -> Tensor: ...

    def hard_done(self, next_state: Tensor, next_goal: Tensor) -> Tensor: ...


@torch.no_grad()
def collect_episode(
    task: TrackingSimulator,
    actor: MotionPolicy,
    config: TrainConfig,
    max_steps: int,
) -> Episode:
    """Log the *executed* noisy action, not the actor's nominal action."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    spaces = actor.spaces
    reference = task.reference(max_steps + config.imagination_horizon + spaces.goal_horizon + 1)
    context = task.sample_context()
    state = task.reset(reference[0], context)
    ideal = task.observation(state)
    states = [state.clone()]
    observations = [ideal.clone()]
    actions: list[Tensor] = []
    rewards: list[Tensor] = []
    dones: list[Tensor] = []

    noisy_history = torch.stack([ideal] * spaces.history)
    noisy_history = noisy_history + task.observation_noise(noisy_history)
    device = next(actor.parameters()).device

    for t in range(max_steps):
        future = reference[t : t + spaces.goal_horizon].unsqueeze(0).to(device)
        action = actor(noisy_history.unsqueeze(0).to(device), future).squeeze(0).cpu()
        executed_action = action + config.action_noise_std * torch.randn_like(action)
        next_state = task.step(executed_action)
        next_ideal = task.observation(next_state)
        reward = task.reward(
            next_state, executed_action, reference[t + 1], previous_state=state
        )
        done = task.hard_done(next_state, reference[t + 1])

        states.append(next_state.clone())
        observations.append(next_ideal.clone())
        actions.append(executed_action.clone())
        rewards.append(reward.clone())
        dones.append(done.clone())

        noisy_observation = next_ideal + task.observation_noise(next_ideal)
        noisy_history = torch.cat((noisy_history[1:], noisy_observation.unsqueeze(0)))
        if bool(done.item()):
            break

    return Episode(
        states=torch.stack(states),
        actions=torch.stack(actions),
        observations=torch.stack(observations),
        rewards=torch.stack(rewards),
        dones=torch.stack(dones),
        context=context.clone(),
        reference=reference,
    )
