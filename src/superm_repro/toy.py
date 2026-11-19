"""A small non-MuJoCo tracking task for verifying the algorithm end to end."""

import math

import torch
from torch import Tensor

from .config import Spaces


class PointMassTracking:
    """Track a moving 2D target with friction-conditioned accelerations."""

    spaces = Spaces(state=6, action=2, context=1, observation=6, goal=2)
    dt = 0.1

    @staticmethod
    def reference(length: int, device: torch.device | str = "cpu") -> Tensor:
        time = torch.arange(length, device=device, dtype=torch.float32) * PointMassTracking.dt
        return torch.stack(
            (0.8 * torch.sin(0.7 * time), 0.8 * torch.sin(0.43 * time + math.pi / 4)),
            dim=-1,
        )

    @staticmethod
    def simulator_step(state: Tensor, action: Tensor, context: Tensor) -> Tensor:
        """A known true transition used only during simulator data collection."""
        velocity = (1 - PointMassTracking.dt * context) * state[..., 2:4]
        velocity = velocity + PointMassTracking.dt * action
        position = state[..., :2] + PointMassTracking.dt * velocity
        return torch.cat((position, velocity, action), dim=-1)

    @staticmethod
    def observation(state: Tensor) -> Tensor:
        return state

    @staticmethod
    def observation_noise(ideal: Tensor) -> Tensor:
        noise = torch.zeros_like(ideal)
        noise[..., :4] = 0.01 * torch.randn_like(ideal[..., :4])
        return noise

    @staticmethod
    def reward(
        next_state: Tensor, action: Tensor, next_goal: Tensor,
        previous_state: Tensor | None = None,
    ) -> Tensor:
        squared_error = (next_state[..., :2] - next_goal).square().sum(dim=-1)
        return torch.exp(-2.0 * squared_error) - 0.01 * action.square().sum(dim=-1)

    @staticmethod
    def hard_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        distance = (next_state[..., :2] - next_goal).norm(dim=-1)
        return (distance > 3.0).to(next_state.dtype)

    @staticmethod
    def soft_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        """A differentiable surrogate for the simulator's hard termination check."""
        distance = (next_state[..., :2] - next_goal).norm(dim=-1)
        return torch.sigmoid(10.0 * (distance - 3.0))
