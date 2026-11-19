"""A genuine MuJoCo cart-pole with an unactuated upright pendulum."""

import math
import random

import numpy as np
import torch
from torch import Tensor

from .config import Spaces


MJCF = """
<mujoco model="superm_repro_cartpole">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="rail" type="box" pos="0 0 0.055" size="2 0.12 0.025"
          rgba="0.25 0.25 0.3 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 0.16">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-1.8 1.8"
             damping="0.1" armature="0.02"/>
      <geom name="cart_box" type="box" size="0.12 0.10 0.06" mass="1.0"
            rgba="0.2 0.45 0.85 1"/>
      <body name="pole" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0"
               damping="0.01" armature="0.001"/>
        <geom name="pole_rod" type="capsule" fromto="0 0 0 0 0 0.8"
              size="0.025" mass="0.15" rgba="0.9 0.5 0.15 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="slider" gear="1"
           ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


class MujocoCartPole:
    """Balance one passive upright pole by applying a horizontal force to its cart.

    State: [cart_position, pole_angle, cart_velocity, pole_velocity, previous_action].
    Goal: [target_cart_position, target_pole_angle]. The default target is upright.
    Action: one normalized horizontal cart force, scaled to Newtons by action_scale.
    Context: randomized generalized joint-damping multiplier.
    """

    spaces = Spaces(state=5, action=1, context=1, observation=5, goal=2)
    control_dt = 0.02
    action_scale = 10.0

    def __init__(self, motion: str = "balance") -> None:
        if motion not in ("balance", "track"):
            raise ValueError("motion must be 'balance' or 'track'")
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("Install the MuJoCo extra: pip install -e '.[sim]'") from error

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.base_damping = self.model.dof_damping.copy()
        self.previous_action = 0.0
        self.motion = motion

    def reference(self, length: int) -> Tensor:
        # The first task is balance. A small cart-position reference is optional.
        goals = torch.zeros(length, 2, dtype=torch.float32)
        if self.motion == "track":
            phase = random.uniform(0.0, 2.0 * math.pi)
            time = torch.arange(length, dtype=torch.float32) * self.control_dt
            goals[:, 0] = 0.2 * torch.sin(0.4 * time + phase)
        return goals

    @staticmethod
    def sample_context() -> Tensor:
        return torch.tensor([random.uniform(0.8, 1.2)], dtype=torch.float32)

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor:
        self.mujoco.mj_resetData(self.model, self.data)
        self.model.dof_damping[:] = self.base_damping * float(context[0])
        self.data.qpos[0] = float(initial_goal[0])
        # The policy must recover from small random pole perturbations.
        self.data.qpos[1] = float(initial_goal[1]) + random.uniform(-0.08, 0.08)
        self.data.qvel[:] = 0.0
        self.data.ctrl[0] = 0.0
        self.previous_action = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def _state(self) -> Tensor:
        values = np.concatenate(
            (self.data.qpos[:2], self.data.qvel[:2], [self.previous_action])
        )
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
        noise = torch.zeros_like(ideal)
        noise[..., :2] = 0.005 * (2.0 * torch.rand_like(ideal[..., :2]) - 1.0)
        noise[..., 2:4] = 0.02 * (2.0 * torch.rand_like(ideal[..., 2:4]) - 1.0)
        return noise

    @staticmethod
    def reward(next_state: Tensor, action: Tensor, next_goal: Tensor) -> Tensor:
        """Differentiable upright and cart-position tracking reward."""
        position_error = next_state[..., 0] - next_goal[..., 0]
        angle_error = next_state[..., 1] - next_goal[..., 1]
        return torch.exp(-position_error.square() - 4.0 * angle_error.square()) - (
            0.005 * action.square().sum(dim=-1)
        )

    @staticmethod
    def hard_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        position_error = (next_state[..., 0] - next_goal[..., 0]).abs()
        angle_error = (next_state[..., 1] - next_goal[..., 1]).abs()
        return ((position_error > 1.5) | (angle_error > 0.55)).to(next_state.dtype)

    @staticmethod
    def soft_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        position_error = (next_state[..., 0] - next_goal[..., 0]).abs()
        angle_error = (next_state[..., 1] - next_goal[..., 1]).abs()
        cart_probability = torch.sigmoid(15.0 * (position_error - 1.5))
        pole_probability = torch.sigmoid(15.0 * (angle_error - 0.55))
        return 1.0 - (1.0 - cart_probability) * (1.0 - pole_probability)
