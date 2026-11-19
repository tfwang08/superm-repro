"""A real MuJoCo two-joint tracking task for the first reproduction milestone."""

import math
import random

import numpy as np
import torch
from torch import Tensor

from .config import Spaces


MJCF = """
<mujoco model="superm_repro_two_joint">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper_arm" pos="0 0 1.0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.4 1.4"
             damping="0.4" armature="0.02"/>
      <geom name="link_1" type="capsule" fromto="0 0 0 0 0 -0.40"
            size="0.045" mass="0.8" rgba="0.2 0.5 0.9 1"/>
      <body name="forearm" pos="0 0 -0.40">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.4 1.4"
               damping="0.3" armature="0.01"/>
        <geom name="link_2" type="capsule" fromto="0 0 0 0 0 -0.35"
              size="0.035" mass="0.5" rgba="0.9 0.6 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pd" joint="shoulder" kp="30" kv="2"
              ctrlrange="-1.3 1.3" ctrllimited="true"/>
    <position name="elbow_pd" joint="elbow" kp="25" kv="1.5"
              ctrlrange="-1.3 1.3" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


class MujocoJointTracking:
    """Control a two-link arm with normalized joint-setpoint actions at 50 Hz."""

    spaces = Spaces(state=6, action=2, context=1, observation=6, goal=2)
    control_dt = 0.02
    action_scale = 1.0

    def __init__(self) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("Install the MuJoCo extra: pip install -e '.[sim]'") from error

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.base_damping = self.model.dof_damping.copy()
        self.previous_action = np.zeros(2, dtype=np.float64)

    @classmethod
    def reference(cls, length: int, phase: float | None = None) -> Tensor:
        if phase is None:
            phase = random.uniform(0, 2 * math.pi)
        time = torch.arange(length, dtype=torch.float32) * cls.control_dt + phase
        return torch.stack((0.35 * torch.sin(time), 0.35 * torch.sin(0.7 * time)), dim=-1)

    @staticmethod
    def sample_context() -> Tensor:
        # Randomized joint damping is passed to the world model as context xi.
        return torch.tensor([random.uniform(0.7, 1.3)], dtype=torch.float32)

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor:
        self.mujoco.mj_resetData(self.model, self.data)
        self.model.dof_damping[:] = self.base_damping * float(context[0])
        self.data.qpos[:2] = initial_goal.cpu().numpy()
        self.data.qvel[:2] = 0.0
        self.data.ctrl[:2] = initial_goal.cpu().numpy()
        self.previous_action[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def _state(self) -> Tensor:
        values = np.concatenate((self.data.qpos[:2], self.data.qvel[:2], self.previous_action))
        return torch.as_tensor(values.copy(), dtype=torch.float32)

    def step(self, executed_action: Tensor) -> Tensor:
        action = executed_action.detach().cpu().numpy()
        self.data.ctrl[:2] = np.clip(
            self.action_scale * action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.previous_action[:] = action
        return self._state()

    @staticmethod
    def observation(state: Tensor) -> Tensor:
        return state

    @staticmethod
    def observation_noise(ideal: Tensor) -> Tensor:
        noise = torch.zeros_like(ideal)
        noise[..., :2] = 0.01 * (2 * torch.rand_like(ideal[..., :2]) - 1)
        noise[..., 2:4] = 0.05 * (2 * torch.rand_like(ideal[..., 2:4]) - 1)
        return noise

    @staticmethod
    def reward(next_state: Tensor, action: Tensor, next_goal: Tensor) -> Tensor:
        position_error = (next_state[..., :2] - next_goal).square().sum(dim=-1)
        return torch.exp(-4.0 * position_error) - 0.01 * action.square().sum(dim=-1)

    @staticmethod
    def hard_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        error = (next_state[..., :2] - next_goal).norm(dim=-1)
        return (error > 1.2).to(next_state.dtype)

    @staticmethod
    def soft_done(next_state: Tensor, next_goal: Tensor) -> Tensor:
        error = (next_state[..., :2] - next_goal).norm(dim=-1)
        return torch.sigmoid(15.0 * (error - 1.2))
