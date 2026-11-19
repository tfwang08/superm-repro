"""MuJoCo carts with one, two, or three passive inverted pendulum links."""

import math
import random

import numpy as np
import torch
from torch import Tensor

from .config import Spaces


def build_cartpole_mjcf(links: int) -> str:
    """Build a serial passive pendulum: only the cart has an actuator."""
    if links not in (1, 2, 3):
        raise ValueError("links must be 1, 2, or 3")
    lengths = (0.8, 0.65, 0.5)
    masses = (0.15, 0.1, 0.07)
    parts = [
        '<mujoco model="superm_repro_multilink_cartpole">',
        '<compiler angle="radian"/>',
        '<option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>',
        '<worldbody>',
        '<geom name="rail" type="box" pos="0 0 0.055" size="2 0.12 0.025" '
        'rgba="0.25 0.25 0.3 1" contype="0" conaffinity="0"/>',
        '<body name="cart" pos="0 0 0.16">',
        '<joint name="slider" type="slide" axis="1 0 0" limited="true" '
        'range="-1.8 1.8" damping="0.1" armature="0.02"/>',
        '<geom name="cart_box" type="box" size="0.12 0.10 0.06" mass="1.0" '
        'rgba="0.2 0.45 0.85 1"/>',
    ]
    for i in range(links):
        pivot = 0.06 if i == 0 else lengths[i - 1]
        parts.extend((
            f'<body name="pole_{i}" pos="0 0 {pivot}">',
            f'<joint name="hinge_{i}" type="hinge" axis="0 1 0" '
            f'damping="{0.01 + 0.005 * i}" armature="0.001"/>',
            f'<geom name="rod_{i}" type="capsule" '
            f'fromto="0 0 0 0 0 {lengths[i]}" size="0.025" '
            f'mass="{masses[i]}" rgba="0.9 0.5 0.15 1"/>',
        ))
    parts.extend("</body>" for _ in range(links + 1))
    parts.extend((
        "</worldbody>",
        '<actuator><motor name="cart_force" joint="slider" gear="1" '
        'ctrllimited="true" ctrlrange="-20 20"/></actuator>',
        "</mujoco>",
    ))
    return "\n".join(parts)


class MujocoCartPole:
    """Control 1-3 hinged rods using horizontal cart force alone.

    State: [cart_x, relative_joint_angles, generalized_velocities, previous_action].
    Goal: [target_cart_x, target_relative_joint_angles].
    """

    control_dt = 0.02
    action_scale = 10.0

    def __init__(self, motion: str = "step", links: int = 1) -> None:
        if motion not in ("balance", "track", "step", "reverse"):
            raise ValueError("motion must be balance, track, step, or reverse")
        if links not in (1, 2, 3):
            raise ValueError("links must be 1, 2, or 3")
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("Install MuJoCo with pip install -e '.[sim]'") from error
        self.motion = motion
        self.links = links
        self.positions = links + 1
        state_dim = 2 * self.positions + 1
        self.spaces = Spaces(
            state=state_dim, action=1, context=1,
            observation=state_dim, goal=self.positions,
        )
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(build_cartpole_mjcf(links))
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.base_damping = self.model.dof_damping.copy()
        self.previous_action = 0.0

    def reference(self, length: int) -> Tensor:
        goals = torch.zeros(length, self.positions, dtype=torch.float32)
        if self.motion == "track":
            phase = random.uniform(0.0, 2.0 * math.pi)
            time = torch.arange(length, dtype=torch.float32) * self.control_dt
            goals[:, 0] = 0.2 * torch.sin(0.4 * time + phase)
        elif self.motion == "step":
            switch = min(random.randint(15, 50), length)
            goals[switch:, 0] = random.uniform(-1.15, 1.15)
        elif self.motion == "reverse":
            sign = random.choice((-1.0, 1.0))
            for start in range(20, length, 64):
                goals[start:, 0] = 0.6 * sign
                sign = -sign
        return goals

    @staticmethod
    def sample_context() -> Tensor:
        return torch.tensor([random.uniform(0.8, 1.2)], dtype=torch.float32)

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor:
        self.mujoco.mj_resetData(self.model, self.data)
        self.model.dof_damping[:] = self.base_damping * float(context[0])
        self.data.qpos[0] = float(initial_goal[0])
        for i in range(self.links):
            amplitude = 0.08 if i == 0 else 0.03
            self.data.qpos[i + 1] = float(initial_goal[i + 1]) + random.uniform(
                -amplitude, amplitude
            )
        self.data.qvel[:] = 0.0
        self.data.ctrl[0] = 0.0
        self.previous_action = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def _state(self) -> Tensor:
        values = np.concatenate((
            self.data.qpos[:self.positions],
            self.data.qvel[:self.positions],
            [self.previous_action],
        ))
        return torch.as_tensor(values.copy(), dtype=torch.float32)

    def step(self, executed_action: Tensor) -> Tensor:
        action = float(executed_action.detach().cpu().reshape(-1)[0])
        low, high = self.model.actuator_ctrlrange[0]
        self.data.ctrl[0] = np.clip(action * self.action_scale, low, high)
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.previous_action = action
        return self._state()

    @staticmethod
    def observation(state: Tensor) -> Tensor:
        return state

    @staticmethod
    def observation_noise(ideal: Tensor) -> Tensor:
        positions = (ideal.shape[-1] - 1) // 2
        noise = torch.zeros_like(ideal)
        noise[..., :positions] = 0.005 * (
            2.0 * torch.rand_like(ideal[..., :positions]) - 1.0
        )
        noise[..., positions:2 * positions] = 0.02 * (
            2.0 * torch.rand_like(ideal[..., positions:2 * positions]) - 1.0
        )
        return noise

    @staticmethod
    def _tracking_errors(state: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        count = goal.shape[-1] - 1
        cart_error = state[..., 0] - goal[..., 0]
        absolute_angles = torch.cumsum(state[..., 1:count + 1], dim=-1)
        absolute_targets = torch.cumsum(goal[..., 1:], dim=-1)
        return cart_error, absolute_angles - absolute_targets

    @classmethod
    def reward(cls, next_state: Tensor, action: Tensor, next_goal: Tensor) -> Tensor:
        cart_error, angle_errors = cls._tracking_errors(next_state, next_goal)
        return torch.exp(
            -cart_error.square() - 4.0 * angle_errors.square().mean(dim=-1)
        ) - 0.005 * action.square().sum(dim=-1)

    @classmethod
    def hard_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        cart_error, angle_errors = cls._tracking_errors(next_state, next_goal)
        return ((cart_error.abs() > 1.5) |
                (angle_errors.abs().amax(dim=-1) > 0.55)).to(next_state.dtype)

    @classmethod
    def soft_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        cart_error, angle_errors = cls._tracking_errors(next_state, next_goal)
        cart_probability = torch.sigmoid(15.0 * (cart_error.abs() - 1.5))
        pole_probability = torch.sigmoid(
            15.0 * (angle_errors.abs().amax(dim=-1) - 0.55)
        )
        return 1.0 - (1.0 - cart_probability) * (1.0 - pole_probability)
