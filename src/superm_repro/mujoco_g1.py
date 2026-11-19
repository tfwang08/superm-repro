"""Contact-rich Unitree G1 standing task for the differentiable training loop.

The physics uses MuJoCo Menagerie's G1 MJX model with the visual meshes removed.
The inertial properties, primitive collision geometries, contacts, and PD joint
actuators are retained. This is a first standing benchmark, not motion imitation.
"""

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import Spaces


G1_SCENE = Path(__file__).resolve().parent / "assets" / "unitree_g1" / "scene_mjx.xml"


class MujocoG1Standing:
    """Control a floating-base 29-actuator G1 by changing PD joint setpoints.

    State: [qpos(36), qvel(35), previously executed action(29)].
    Goal: [pelvis height(1), joint reference positions(29)].
    Action: normalized offset from the nominal standing joint configuration.
    Context: foot-contact friction multiplier, fixed throughout each episode.
    """

    control_dt = 0.02
    action_scale = 0.35
    qpos_size = 36
    qvel_size = 35
    actuator_size = 29

    def __init__(self, scene_path: str | Path | None = None) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("Install MuJoCo with pip install -e '.[sim]'") from error

        self.mujoco = mujoco
        scene = G1_SCENE if scene_path is None else Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (36, 35, 29):
            raise ValueError("Expected a floating-base G1 with 29 position actuators")
        actuator_joint_addresses = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        if not np.array_equal(actuator_joint_addresses, np.arange(7, 36)):
            raise ValueError("G1 actuators must follow the joint order in qpos")
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        if not math.isclose(self.substeps * self.model.opt.timestep, self.control_dt):
            raise ValueError("The MuJoCo timestep must divide the control period")
        self.stand_keyframe = self.model.key("home").id
        self.stand_pose = self.model.key("home").qpos.copy()
        self.stand_ctrl = self.model.key("home").ctrl.copy()
        self.base_pair_friction = self.model.pair_friction.copy()
        self.previous_action = np.zeros(self.actuator_size, dtype=np.float64)
        self.spaces = Spaces(
            state=self.qpos_size + self.qvel_size + self.actuator_size,
            action=self.actuator_size,
            context=1,
            observation=self.qpos_size + self.qvel_size + self.actuator_size,
            goal=1 + self.actuator_size,
        )

    def reference(self, length: int) -> Tensor:
        if length < 1:
            raise ValueError("Reference length must be positive")
        stand_goal = np.concatenate(([self.stand_pose[2]], self.stand_ctrl))
        return torch.as_tensor(stand_goal.copy(), dtype=torch.float32).repeat(length, 1)

    @staticmethod
    def sample_context() -> Tensor:
        return torch.tensor([random.uniform(0.8, 1.2)], dtype=torch.float32)

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor:
        if initial_goal.shape != (self.spaces.goal,) or context.shape != (1,):
            raise ValueError("Unexpected G1 goal or context shape")
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_keyframe)
        friction_scale = float(context[0])
        self.model.pair_friction[:] = self.base_pair_friction
        self.model.pair_friction[:, :2] *= friction_scale

        self.data.qpos[:2] += np.array([random.uniform(-0.015, 0.015) for _ in range(2)])
        self.data.qpos[2] = float(initial_goal[0])
        roll = random.uniform(-0.035, 0.035)
        pitch = random.uniform(-0.035, 0.035)
        self.data.qpos[3:7] = (
            math.cos(roll / 2) * math.cos(pitch / 2),
            math.sin(roll / 2) * math.cos(pitch / 2),
            math.cos(roll / 2) * math.sin(pitch / 2),
            math.sin(roll / 2) * math.sin(pitch / 2),
        )
        self.data.qpos[7:] = np.asarray(initial_goal[1:]) + np.array(
            [random.uniform(-0.015, 0.015) for _ in range(self.actuator_size)]
        )
        self.data.qvel[:] = [random.uniform(-0.02, 0.02) for _ in range(self.qvel_size)]
        self.data.ctrl[:] = np.clip(
            np.asarray(initial_goal[1:]),
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.previous_action[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def _state(self) -> Tensor:
        values = np.concatenate((self.data.qpos, self.data.qvel, self.previous_action))
        return torch.as_tensor(values.copy(), dtype=torch.float32)

    def step(self, executed_action: Tensor) -> Tensor:
        action = np.asarray(executed_action.detach().cpu(), dtype=np.float64).reshape(-1)
        if action.shape != (self.actuator_size,):
            raise ValueError("G1 action must contain 29 joint setpoint offsets")
        target = self.stand_ctrl + self.action_scale * action
        self.data.ctrl[:] = np.clip(
            target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
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
        noise[..., :36] = 0.002 * (2.0 * torch.rand_like(ideal[..., :36]) - 1.0)
        noise[..., 36:71] = 0.01 * (2.0 * torch.rand_like(ideal[..., 36:71]) - 1.0)
        return noise

    @classmethod
    def _standing_errors(cls, state: Tensor, goal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # The horizontal components of a unit quaternion encode gravity tilt.
        tilt = 2.0 * torch.linalg.vector_norm(state[..., 4:6], dim=-1)
        height_error = state[..., 2] - goal[..., 0]
        horizontal_error = torch.linalg.vector_norm(state[..., :2], dim=-1)
        return tilt, height_error, horizontal_error

    @classmethod
    def reward(cls, next_state: Tensor, action: Tensor, next_goal: Tensor) -> Tensor:
        tilt, height_error, horizontal_error = cls._standing_errors(next_state, next_goal)
        joint_error = next_state[..., 7:36] - next_goal[..., 1:]
        base_velocity = next_state[..., 36:42]
        posture_cost = (
            6.0 * tilt.square()
            + 20.0 * height_error.square()
            + 2.0 * horizontal_error.square()
            + joint_error.square().mean(dim=-1)
            + 0.03 * base_velocity.square().sum(dim=-1)
        )
        return torch.exp(-posture_cost) - 0.002 * action.square().mean(dim=-1)

    @classmethod
    def hard_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        tilt, _, horizontal_error = cls._standing_errors(next_state, next_goal)
        return (
            (next_state[..., 2] < 0.55) | (tilt > 0.7) | (horizontal_error > 0.75)
        ).to(next_state.dtype)

    @classmethod
    def soft_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        tilt, _, horizontal_error = cls._standing_errors(next_state, next_goal)
        height_probability = torch.sigmoid(20.0 * (0.55 - next_state[..., 2]))
        tilt_probability = torch.sigmoid(12.0 * (tilt - 0.7))
        drift_probability = torch.sigmoid(12.0 * (horizontal_error - 0.75))
        return 1.0 - (
            (1.0 - height_probability)
            * (1.0 - tilt_probability)
            * (1.0 - drift_probability)
        )
