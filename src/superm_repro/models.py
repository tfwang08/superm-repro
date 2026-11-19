"""Differentiable dynamics, reference-conditioned actor, and bootstrap critic."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .config import Spaces


def make_mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    """Construct an ELU MLP with a linear output layer."""
    layers: list[nn.Module] = []
    dim = input_dim
    for width in hidden:
        layers.extend((nn.Linear(dim, width), nn.ELU()))
        dim = width
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """Learn f(s, a, context) -> next state; task heads may be analytic."""

    def __init__(self, spaces: Spaces, hidden: Sequence[int] = (2048, 2048, 2048)):
        super().__init__()
        self.dynamics = make_mlp(
            spaces.state + spaces.action + spaces.context, hidden, spaces.state
        )

    def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
        state_delta = self.dynamics(torch.cat((state, action, context), dim=-1))
        return state + state_delta


class MotionPolicy(nn.Module):
    """Encode 10 future targets and decode actions from 10 observed frames."""

    def __init__(
        self,
        spaces: Spaces,
        latent_dim: int = 32,
        encoder_hidden: Sequence[int] = (2048, 1024, 512, 256),
        decoder_hidden: Sequence[int] = (2048, 1024, 512, 256),
    ):
        super().__init__()
        self.spaces = spaces
        self.goal_encoder = make_mlp(
            spaces.goal_horizon * spaces.goal, encoder_hidden, latent_dim
        )
        self.action_decoder = make_mlp(
            spaces.history * spaces.observation + latent_dim,
            decoder_hidden,
            spaces.action,
        )

    def forward(self, observation_history: Tensor, future_goals: Tensor) -> Tensor:
        if observation_history.shape[-2:] != (
            self.spaces.history,
            self.spaces.observation,
        ):
            raise ValueError("Unexpected observation history shape")
        if future_goals.shape[-2:] != (self.spaces.goal_horizon, self.spaces.goal):
            raise ValueError("Unexpected future reference shape")
        latent = torch.tanh(self.goal_encoder(future_goals.flatten(start_dim=-2)))
        inputs = torch.cat((observation_history.flatten(start_dim=-2), latent), dim=-1)
        return torch.tanh(self.action_decoder(inputs))


class Critic(nn.Module):
    """Estimate the discounted return after the current reference-conditioned state."""

    def __init__(self, spaces: Spaces, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.spaces = spaces
        self.value_net = make_mlp(
            spaces.state + spaces.goal_horizon * spaces.goal + spaces.context,
            hidden,
            1,
        )

    def forward(self, state: Tensor, future_goals: Tensor, context: Tensor) -> Tensor:
        inputs = torch.cat((state, future_goals.flatten(start_dim=-2), context), dim=-1)
        return self.value_net(inputs).squeeze(-1)
