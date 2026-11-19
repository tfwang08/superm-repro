"""The reference-relative antisymmetric policy enforces cart symmetries exactly."""

import torch

from superm_repro.config import Spaces
from superm_repro.models import MotionPolicy


def make_policy(links: int = 1) -> MotionPolicy:
    positions = links + 1
    spaces = Spaces(
        state=2 * positions + 1, observation=2 * positions + 1,
        action=1, context=1, goal=positions,
    )
    return MotionPolicy(
        spaces, encoder_hidden=(24,), decoder_hidden=(24,),
        relative_goal=True, antisymmetric=True,
    )


def test_cart_translation_invariance() -> None:
    torch.manual_seed(11)
    policy = make_policy()
    observations = torch.randn(5, 10, 5)
    goals = torch.randn(5, 10, 2)
    shifted_observations = observations.clone()
    shifted_goals = goals.clone()
    shifted_observations[..., 0] += 0.6
    shifted_goals[..., 0] += 0.6
    torch.testing.assert_close(
        policy(observations, goals),
        policy(shifted_observations, shifted_goals),
        atol=1e-6, rtol=1e-6,
    )


def test_left_right_antisymmetry_and_gradients() -> None:
    torch.manual_seed(13)
    policy = make_policy(links=3)
    observations = torch.randn(4, 10, 9)
    goals = torch.randn(4, 10, 4)
    forward = policy(observations, goals)
    mirrored = policy(-observations, -goals)
    torch.testing.assert_close(forward, -mirrored, atol=1e-6, rtol=1e-6)
    forward.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
