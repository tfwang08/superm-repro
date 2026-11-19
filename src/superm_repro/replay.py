"""Trajectory replay with contiguous world-model windows and imagination starts."""

import random
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Episode:
    """One simulator segment with reference targets beyond the segment boundary."""

    states: Tensor  # [T+1, state_dim]
    actions: Tensor  # [T, action_dim], including collection-time action noise
    observations: Tensor  # [T+1, observation_dim], without sensor noise
    rewards: Tensor  # [T], reward for each transition t -> t+1
    dones: Tensor  # [T], hard simulator termination labels
    context: Tensor  # [context_dim], held fixed within an episode
    reference: Tensor  # [T+H+K, goal_dim], independent of early termination

    def __post_init__(self) -> None:
        steps = self.actions.shape[0]
        if steps < 1:
            raise ValueError("An episode must contain at least one transition")
        if self.states.shape[0] != steps + 1 or self.observations.shape[0] != steps + 1:
            raise ValueError("States and observations must have T+1 entries")
        if self.rewards.shape != (steps,) or self.dones.shape != (steps,):
            raise ValueError("Rewards and dones must have T entries")
        if self.context.ndim != 1 or self.reference.ndim != 2:
            raise ValueError("Expected one context vector and a sequence of reference frames")
        if any(x.requires_grad for x in self.__dict__.values()):
            raise ValueError("Replay must contain detached simulator data")

    @property
    def length(self) -> int:
        return self.actions.shape[0]


@dataclass(frozen=True)
class WindowBatch:
    states: Tensor  # [B, k+1, state_dim]
    actions: Tensor  # [B, k, action_dim]
    observations: Tensor  # [B, k+1, observation_dim]
    rewards: Tensor  # [B, k]
    dones: Tensor  # [B, k]
    contexts: Tensor  # [B, context_dim]

    def to(self, device: torch.device | str) -> "WindowBatch":
        return WindowBatch(*(value.to(device) for value in self.__dict__.values()))


@dataclass(frozen=True)
class StartBatch:
    states: Tensor  # [B, state_dim]
    observation_history: Tensor  # [B, history, observation_dim]
    references: Tensor  # [B, H+K, goal_dim]
    contexts: Tensor  # [B, context_dim]

    def to(self, device: torch.device | str) -> "StartBatch":
        return StartBatch(*(value.to(device) for value in self.__dict__.values()))


class ReplayBuffer:
    """A bounded collection of trajectory segments, preserving transition ordering."""

    def __init__(self, capacity_steps: int):
        if capacity_steps < 1:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = capacity_steps
        self.episodes: deque[Episode] = deque()
        self.steps = 0

    def add(self, episode: Episode) -> None:
        if episode.length > self.capacity_steps:
            raise ValueError("One episode exceeds the replay buffer capacity")
        self.episodes.append(episode)
        self.steps += episode.length
        while self.steps > self.capacity_steps:
            self.steps -= self.episodes.popleft().length

    @staticmethod
    def _pick(eligible: list[tuple[Episode, int]]) -> tuple[Episode, int]:
        episode = random.choices(
            [item[0] for item in eligible], weights=[item[1] for item in eligible], k=1
        )[0]
        number_of_starts = next(count for item, count in eligible if item is episode)
        return episode, random.randrange(number_of_starts)

    def sample_windows(self, batch_size: int, length: int) -> WindowBatch:
        """Draw contiguous k-step paths; never cross episode boundaries."""
        if length < 1 or batch_size < 1:
            raise ValueError("Batch size and window length must be positive")
        eligible = [(ep, ep.length - length + 1) for ep in self.episodes if ep.length >= length]
        if not eligible:
            raise ValueError("No episodes long enough for a model window")
        examples = [self._pick(eligible) for _ in range(batch_size)]
        return WindowBatch(
            states=torch.stack([ep.states[t : t + length + 1] for ep, t in examples]),
            actions=torch.stack([ep.actions[t : t + length] for ep, t in examples]),
            observations=torch.stack(
                [ep.observations[t : t + length + 1] for ep, t in examples]
            ),
            rewards=torch.stack([ep.rewards[t : t + length] for ep, t in examples]),
            dones=torch.stack([ep.dones[t : t + length] for ep, t in examples]),
            contexts=torch.stack([ep.context for ep, _ in examples]),
        )

    def sample_starts(
        self, batch_size: int, horizon: int, history: int, goal_horizon: int
    ) -> StartBatch:
        """Pair logged start states with future goals from the independent reference."""
        if min(batch_size, horizon, history, goal_horizon) < 1:
            raise ValueError("All sampling sizes must be positive")
        reference_span = horizon + goal_horizon
        eligible = [
            (ep, min(ep.length, ep.reference.shape[0] - reference_span + 1))
            for ep in self.episodes
            if ep.reference.shape[0] >= reference_span
        ]
        eligible = [(ep, count) for ep, count in eligible if count > 0]
        if not eligible:
            raise ValueError("No reference trajectories long enough for imagination")
        examples = [self._pick(eligible) for _ in range(batch_size)]
        history_sequences = [
            ep.observations[
                torch.arange(t - history + 1, t + 1).clamp(min=0)
            ]
            for ep, t in examples
        ]
        return StartBatch(
            states=torch.stack([ep.states[t] for ep, t in examples]),
            observation_history=torch.stack(history_sequences),
            references=torch.stack(
                [ep.reference[t : t + reference_span] for ep, t in examples]
            ),
            contexts=torch.stack([ep.context for ep, _ in examples]),
        )
