"""Configuration shared by the reference algorithm and toy smoke test."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Spaces:
    state: int
    action: int
    context: int
    observation: int
    goal: int
    history: int = 10
    goal_horizon: int = 10


@dataclass(frozen=True)
class TrainConfig:
    # Section 3 and Appendix D of arXiv:2610.28480.
    model_window: int = 8
    imagination_horizon: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise_std: float = 0.1
    learning_rate: float = 2e-4
    model_grad_clip: float = 1.0
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 1.0
