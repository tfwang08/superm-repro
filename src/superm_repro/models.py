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

    def __init__(
        self,
        spaces: Spaces,
        hidden: Sequence[int] = (2048, 2048, 2048),
        quaternion_slice: tuple[int, int] | None = None,
        action_state_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        if quaternion_slice is not None and (
            quaternion_slice[1] - quaternion_slice[0] != 4
            or not (0 <= quaternion_slice[0] < quaternion_slice[1] <= spaces.state)
        ):
            raise ValueError("Quaternion slice must identify four state components")
        self.quaternion_slice = quaternion_slice
        if action_state_slice is not None and (
            action_state_slice[1] - action_state_slice[0] != spaces.action
            or not (0 <= action_state_slice[0] < action_state_slice[1] <= spaces.state)
        ):
            raise ValueError("Action-history slice must match the action dimension")
        self.action_state_slice = action_state_slice
        self.dynamics = make_mlp(
            spaces.state + spaces.action + spaces.context, hidden, spaces.state
        )

    def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
        state_delta = self.dynamics(torch.cat((state, action, context), dim=-1))
        prediction = state + state_delta
        if self.quaternion_slice is not None:
            start, end = self.quaternion_slice
            quaternion = prediction[..., start:end]
            quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            prediction = torch.cat(
                (prediction[..., :start], quaternion, prediction[..., end:]), dim=-1
            )
        if self.action_state_slice is not None:
            start, end = self.action_state_slice
            prediction = torch.cat((prediction[..., :start], action, prediction[..., end:]), dim=-1)
        return prediction


class MotionPolicy(nn.Module):
    """Encode 10 future targets and decode actions from 10 observed frames."""

    def __init__(
        self,
        spaces: Spaces,
        latent_dim: int = 32,
        encoder_hidden: Sequence[int] = (2048, 1024, 512, 256),
        decoder_hidden: Sequence[int] = (2048, 1024, 512, 256),
        relative_goal: bool = False,
        antisymmetric: bool = False,
    ):
        super().__init__()
        if antisymmetric and spaces.action != 1:
            raise ValueError("Antisymmetric cart-force policies require one action dimension")
        self.spaces = spaces
        self.relative_goal = relative_goal
        self.antisymmetric = antisymmetric
        self.goal_encoder = make_mlp(spaces.goal_horizon * spaces.goal, encoder_hidden, latent_dim)
        self.action_decoder = make_mlp(
            spaces.history * spaces.observation + latent_dim,
            decoder_hidden,
            spaces.action,
        )

    def _direct_action(self, observation_history: Tensor, future_goals: Tensor) -> Tensor:
        latent = torch.tanh(self.goal_encoder(future_goals.flatten(start_dim=-2)))
        inputs = torch.cat((observation_history.flatten(start_dim=-2), latent), dim=-1)
        return torch.tanh(self.action_decoder(inputs))

    def forward(self, observation_history: Tensor, future_goals: Tensor) -> Tensor:
        if observation_history.shape[-2:] != (
            self.spaces.history,
            self.spaces.observation,
        ):
            raise ValueError("Unexpected observation history shape")
        if future_goals.shape[-2:] != (self.spaces.goal_horizon, self.spaces.goal):
            raise ValueError("Unexpected future reference shape")
        if self.relative_goal:
            # Translation-equivariant representation for the cart coordinate.
            current_target = future_goals[:, 0, :1].unsqueeze(1)
            observation_history = torch.cat(
                (
                    observation_history[..., :1] - current_target,
                    observation_history[..., 1:],
                ),
                dim=-1,
            )
            future_goals = torch.cat(
                (future_goals[..., :1] - current_target, future_goals[..., 1:]),
                dim=-1,
            )
        if self.antisymmetric:
            # f(x) = (h(x) - h(-x))/2 gives f(-x) = -f(x) by construction.
            return 0.5 * (
                self._direct_action(observation_history, future_goals)
                - self._direct_action(-observation_history, -future_goals)
            )
        return self._direct_action(observation_history, future_goals)


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
