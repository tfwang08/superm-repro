"""Deterministic simulator evaluation and short open-loop dynamics diagnostics."""

import random
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .models import MotionPolicy, WorldModel


class EvaluatableTask(Protocol):
    """Common simulator interface for cart-poles and humanoid benchmarks."""

    spaces: object

    def reference(self, length: int) -> Tensor: ...

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor: ...

    def step(self, executed_action: Tensor) -> Tensor: ...

    def observation(self, state: Tensor) -> Tensor: ...

    def reward(
        self, next_state: Tensor, action: Tensor, next_goal: Tensor,
        previous_state: Tensor | None = None,
    ) -> Tensor: ...

    def hard_done(self, next_state: Tensor, next_goal: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class EvaluationStats:
    mean_survival_steps: float
    mean_normalized_return: float
    failure_rate: float
    model_state_mse: float


@torch.no_grad()
def evaluate_tracking_task(
    task: EvaluatableTask,
    actor: MotionPolicy,
    model: WorldModel,
    *,
    episodes: int = 8,
    max_steps: int = 250,
    model_window: int = 8,
    seed: int = 2026,
) -> EvaluationStats:
    """Evaluate identical noiseless starts; count rewards after termination as zero.

    The model metric measures k-step open-loop state MSE on held-out MuJoCo
    transitions using executed actions, with only s_0 taken from simulation.
    """
    if min(episodes, max_steps, model_window) < 1:
        raise ValueError("Evaluation sizes must be positive")
    actor_device = next(actor.parameters()).device
    model_device = next(model.parameters()).device
    saved_random_state = random.getstate()
    total_survival = 0
    total_return = 0.0
    failures = 0
    model_errors: list[float] = []

    try:
        random.seed(seed)
        for _ in range(episodes):
            reference = task.reference(max_steps + actor.spaces.goal_horizon + 1)
            context_factory = getattr(task, "evaluation_context", None)
            context = (
                context_factory() if callable(context_factory)
                else torch.ones(task.spaces.context)
            )
            state = task.reset(reference[0], context)
            history = task.observation(state).repeat(actor.spaces.history, 1)
            states = [state]
            actions = []
            episode_return = 0.0
            terminated = False

            for t in range(max_steps):
                future = reference[t : t + actor.spaces.goal_horizon].unsqueeze(0)
                action = actor(
                    history.unsqueeze(0).to(actor_device), future.to(actor_device)
                ).squeeze(0).cpu()
                next_state = task.step(action)
                episode_return += float(task.reward(
                    next_state, action, reference[t + 1], previous_state=state
                ))
                terminated = bool(task.hard_done(next_state, reference[t + 1]))
                states.append(next_state)
                actions.append(action)
                history = torch.cat((history[1:], task.observation(next_state).unsqueeze(0)))
                state = next_state
                if terminated:
                    break

            total_survival += len(actions)
            total_return += episode_return / max_steps
            failures += int(terminated)

            for start in range(0, len(actions) - model_window + 1, model_window):
                predicted = states[start].unsqueeze(0).to(model_device)
                context_batch = context.unsqueeze(0).to(model_device)
                for offset in range(model_window):
                    executed_action = actions[start + offset].unsqueeze(0).to(model_device)
                    predicted = model(predicted, executed_action, context_batch)
                    target = states[start + offset + 1].unsqueeze(0).to(model_device)
                    model_errors.append((predicted - target).square().mean().item())
    finally:
        random.setstate(saved_random_state)

    return EvaluationStats(
        mean_survival_steps=total_survival / episodes,
        mean_normalized_return=total_return / episodes,
        failure_rate=failures / episodes,
        model_state_mse=(sum(model_errors) / len(model_errors)) if model_errors else float("nan"),
    )


# Preserve the previous public API used by cart-pole experiments and tests.
evaluate_cartpole = evaluate_tracking_task
