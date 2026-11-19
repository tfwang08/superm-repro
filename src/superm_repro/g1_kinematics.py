"""Differentiable rigid-body forward kinematics for the bundled G1 model.

MuJoCo provides the simulator targets, but policy gradients must not depend on
autograd through MuJoCo. All body poses and spatial velocities in this module
are computed from state tensors with PyTorch operations.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product for scalar-first unit quaternions."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Rotate 3D vectors with scalar-first unit quaternions."""
    imaginary = quaternion[..., 1:]
    cross = 2.0 * torch.linalg.cross(imaginary, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.linalg.cross(imaginary, cross)


def quat_angle(a: Tensor, b: Tensor) -> Tensor:
    """Differentiable shortest SO(3) orientation distance in radians."""
    conjugate = torch.cat((a[..., :1], -a[..., 1:]), dim=-1)
    difference = quat_mul(conjugate, b)
    return 2.0 * torch.atan2(
        difference[..., 1:].norm(dim=-1), difference[..., 0].abs().clamp_min(1e-8)
    )


@dataclass(frozen=True)
class BodyKinematics:
    positions: Tensor  # [..., bodies, 3]
    quaternions: Tensor  # [..., bodies, 4]
    linear_velocities: Tensor  # [..., bodies, 3], body-origin velocity in world
    angular_velocities: Tensor  # [..., bodies, 3], world frame


class G1Kinematics:
    """Pure PyTorch FK for the 29-hinge, free-root MuJoCo Menagerie G1."""

    def __init__(self, model: object) -> None:
        self.parents = tuple(int(v) for v in model.body_parentid)
        self.body_positions = torch.tensor(model.body_pos.copy(), dtype=torch.float32)
        self.body_quaternions = torch.tensor(model.body_quat.copy(), dtype=torch.float32)
        self.joint_axes = torch.tensor(model.jnt_axis.copy(), dtype=torch.float32)
        self.joint_qpos = tuple(int(v) for v in model.jnt_qposadr)
        self.joint_qvel = tuple(int(v) for v in model.jnt_dofadr)
        self.body_joints = tuple(
            tuple(
                range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b]))
            )
            for b in range(model.nbody)
        )
        if not np.allclose(model.jnt_pos, 0):
            raise ValueError("The current G1 FK assumes zero local joint-pivot offsets")
        if len(self.parents) != 31 or self.body_joints[1] != (0,):
            raise ValueError("Expected the 31-body Menagerie G1 with a free root")
        if any(len(self.body_joints[b]) > 1 for b in range(2, len(self.parents))):
            raise ValueError("Expected at most one revolute joint per articulated body")

    def forward(self, state: Tensor) -> BodyKinematics:
        leading = state.shape[:-1]
        zeros = state.new_zeros(*leading, 3)
        identity = torch.cat((state.new_ones(*leading, 1), zeros), dim=-1)
        positions = [zeros]
        quaternions = [identity]
        linear_velocities = [zeros]
        angular_velocities = [zeros]

        root_q = state[..., 3:7]
        positions.append(state[..., :3])
        quaternions.append(root_q)
        linear_velocities.append(state[..., 36:39])
        angular_velocities.append(quat_rotate(root_q, state[..., 39:42]))

        for body in range(2, len(self.parents)):
            parent = self.parents[body]
            parent_q = quaternions[parent]
            local_position = self.body_positions[body].to(state)
            local_quaternion = self.body_quaternions[body].to(state)
            offset = quat_rotate(parent_q, local_position.expand_as(positions[parent]))
            position = positions[parent] + offset
            quaternion = quat_mul(parent_q, local_quaternion.expand_as(parent_q))
            angular_velocity = angular_velocities[parent]
            linear_velocity = linear_velocities[parent] + torch.linalg.cross(
                angular_velocity, offset, dim=-1
            )

            for joint in self.body_joints[body]:
                axis = self.joint_axes[joint].to(state)
                angle = state[..., self.joint_qpos[joint]]
                velocity = state[..., 36 + self.joint_qvel[joint]]
                axis_world = quat_rotate(quaternion, axis.expand_as(position))
                half_angle = angle * 0.5
                rotation = torch.cat(
                    (half_angle.cos().unsqueeze(-1), half_angle.sin().unsqueeze(-1) * axis),
                    dim=-1,
                )
                quaternion = quat_mul(quaternion, rotation)
                angular_velocity = angular_velocity + axis_world * velocity.unsqueeze(-1)

            positions.append(position)
            quaternions.append(quaternion)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)

        return BodyKinematics(
            positions=torch.stack(positions[1:], dim=-2),
            quaternions=torch.stack(quaternions[1:], dim=-2),
            linear_velocities=torch.stack(linear_velocities[1:], dim=-2),
            angular_velocities=torch.stack(angular_velocities[1:], dim=-2),
        )
