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
from .g1_kinematics import G1Kinematics, quat_angle

G1_SCENE = Path(__file__).resolve().parent / "assets" / "unitree_g1" / "scene_mjx.xml"


class MujocoG1Standing:
    """Control a floating-base 29-actuator G1 by changing PD joint setpoints.

    State: [qpos(36), qvel(35), previously executed action(29)].
    Goal: [pelvis height(1), joint reference positions(29)].
    Action: normalized offset from the nominal standing joint configuration.
    Context: [foot friction(1), joint calibration(29), torso COM offset(3)].
    """

    control_dt = 0.02
    action_scale = 0.35
    qpos_size = 36
    qvel_size = 35
    actuator_size = 29
    context_size = 33

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
        self.foot_floor_pairs = np.array(
            [
                i
                for i in range(self.model.npair)
                if "_floor" in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_PAIR, i) or "")
            ],
            dtype=np.int32,
        )
        if len(self.foot_floor_pairs) < 2:
            raise ValueError("Expected explicitly paired G1 foot-ground contacts")
        self.torso_body_id = self.model.body("torso_link").id
        self.base_torso_ipos = self.model.body_ipos[self.torso_body_id].copy()
        self.joint_limits = torch.as_tensor(
            self.model.actuator_ctrlrange.copy(), dtype=torch.float32
        )
        self.calibration = np.zeros(self.actuator_size, dtype=np.float64)
        self.previous_action = np.zeros(self.actuator_size, dtype=np.float64)
        self.kinematics = G1Kinematics(self.model)
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_keyframe)
        self.mujoco.mj_forward(self.model, self.data)
        self.target_body_positions = torch.as_tensor(self.data.xpos[1:].copy(), dtype=torch.float32)
        self.target_body_quaternions = torch.as_tensor(
            self.data.xquat[1:].copy(), dtype=torch.float32
        )
        self.target_root_quaternion = torch.as_tensor(
            self.stand_pose[3:7].copy(), dtype=torch.float32
        )
        self.spaces = Spaces(
            state=self.qpos_size + self.qvel_size + self.actuator_size,
            action=self.actuator_size,
            context=self.context_size,
            observation=93,
            goal=1 + self.actuator_size,
        )

    def reference(self, length: int) -> Tensor:
        if length < 1:
            raise ValueError("Reference length must be positive")
        stand_goal = np.concatenate(([self.stand_pose[2]], self.stand_ctrl))
        return torch.as_tensor(stand_goal.copy(), dtype=torch.float32).repeat(length, 1)

    @staticmethod
    def sample_context() -> Tensor:
        return torch.tensor(
            [random.uniform(0.3, 1.2)]
            + [random.uniform(-0.01, 0.01) for _ in range(29)]
            + [
                random.uniform(-0.025, 0.025),
                random.uniform(-0.05, 0.05),
                random.uniform(-0.05, 0.05),
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def evaluation_context() -> Tensor:
        """Use nominal calibration and COM with median foot friction in evaluation."""
        return torch.cat((torch.tensor([0.75]), torch.zeros(32)))

    def reset(self, initial_goal: Tensor, context: Tensor) -> Tensor:
        if initial_goal.shape != (self.spaces.goal,) or context.shape != (self.context_size,):
            raise ValueError("Unexpected G1 goal or context shape")
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_keyframe)
        friction = float(context[0])
        self.model.pair_friction[:] = self.base_pair_friction
        self.model.pair_friction[self.foot_floor_pairs, :2] = friction
        self.calibration[:] = np.asarray(context[1:30])
        self.model.body_ipos[self.torso_body_id] = self.base_torso_ipos + np.asarray(context[30:33])
        self.mujoco.mj_setConst(self.model, self.data)

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
            np.asarray(initial_goal[1:]) + self.calibration,
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
        target = self.stand_ctrl + self.calibration + self.action_scale * action
        self.data.ctrl[:] = np.clip(
            target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.previous_action[:] = action
        return self._state()

    def observation(self, state: Tensor) -> Tensor:
        """Appendix A.1: proprioception only, excluding global root position/pose."""
        w, x, y, z = state[..., 3:7].unbind(dim=-1)
        projected_gravity = torch.stack(
            (2.0 * (w * y - x * z), -2.0 * (y * z + w * x), -1.0 + 2.0 * (x.square() + y.square())),
            dim=-1,
        )
        return torch.cat(
            (
                state[..., 7:36] - self.reference_pose(state),
                state[..., 42:71],
                state[..., 39:42],
                projected_gravity,
                state[..., 71:100],
            ),
            dim=-1,
        )

    def reference_pose(self, state: Tensor) -> Tensor:
        return torch.as_tensor(self.stand_ctrl, dtype=state.dtype, device=state.device)

    @staticmethod
    def observation_noise(ideal: Tensor) -> Tensor:
        noise = torch.zeros_like(ideal)
        for start, end, amplitude in (
            (0, 29, 0.01),
            (29, 58, 0.5),
            (58, 61, 0.2),
            (61, 64, 0.05),
        ):
            noise[..., start:end] = amplitude * (2.0 * torch.rand_like(ideal[..., start:end]) - 1.0)
        return noise

    @staticmethod
    def _standing_errors(state: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        # For an upright reference, tilt error is the change in local gravity z.
        tilt = 2.0 * state[..., 4:6].square().sum(dim=-1)
        height_error = state[..., 2] - goal[..., 0]
        return tilt, height_error

    def reward(
        self,
        next_state: Tensor,
        action: Tensor,
        next_goal: Tensor,
        previous_state: Tensor | None = None,
    ) -> Tensor:
        """Appendix A.2 body tracking and regularization on the standing reference."""
        kinematics = self.kinematics.forward(next_state)
        desired_positions = self.target_body_positions.to(next_state)
        desired_quaternions = self.target_body_quaternions.to(next_state)
        body_position_error = (kinematics.positions - desired_positions).square().sum(dim=-1)
        body_orientation_error = quat_angle(kinematics.quaternions, desired_quaternions).square()
        linear_velocity_error = kinematics.linear_velocities.square().sum(dim=-1)
        angular_velocity_error = kinematics.angular_velocities.square().sum(dim=-1)
        body_score = (
            torch.exp(-body_position_error.mean(dim=-1) / 0.3**2)
            + torch.exp(-body_orientation_error.mean(dim=-1) / 0.4**2)
            + torch.exp(-linear_velocity_error.mean(dim=-1) / 1.0**2)
            + torch.exp(-angular_velocity_error.mean(dim=-1) / 3.142**2)
        )

        desired_root_position = torch.stack(
            (
                torch.zeros_like(next_goal[..., 0]),
                torch.zeros_like(next_goal[..., 0]),
                next_goal[..., 0],
            ),
            dim=-1,
        )
        root_position_error = (next_state[..., :3] - desired_root_position).square().sum(-1)
        root_orientation_error = quat_angle(
            next_state[..., 3:7], self.target_root_quaternion.to(next_state)
        ).square()
        root_score = 0.5 * torch.exp(-root_position_error / 0.3**2) + 0.5 * torch.exp(
            -root_orientation_error / 0.4**2
        )

        previous_action = (
            previous_state[..., -self.actuator_size :]
            if previous_state is not None
            else torch.zeros_like(action)
        )
        action_smoothness = (action - previous_action).square().sum(dim=-1)
        limits = self.joint_limits.to(next_state)
        joint_positions = next_state[..., 7:36]
        limit_penalty = (
            torch.relu(joint_positions - limits[:, 1]) + torch.relu(limits[:, 0] - joint_positions)
        ).sum(dim=-1)
        return self.control_dt * (
            body_score + root_score - 0.1 * action_smoothness - 2.0 * limit_penalty
        )

    @classmethod
    def hard_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        tilt, height_error = cls._standing_errors(next_state, next_goal)
        return ((height_error.abs() > 0.25) | (tilt > 0.8)).to(next_state.dtype)

    @classmethod
    def soft_done(cls, next_state: Tensor, next_goal: Tensor) -> Tensor:
        tilt, height_error = cls._standing_errors(next_state, next_goal)
        height_probability = torch.sigmoid(20.0 * (height_error.abs() - 0.25))
        tilt_probability = torch.sigmoid(15.0 * (tilt - 0.8))
        return 1.0 - (1.0 - height_probability) * (1.0 - tilt_probability)
