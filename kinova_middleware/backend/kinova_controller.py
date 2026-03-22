from __future__ import annotations

from typing import Any, Sequence

from kinova_backend import CartesianPose, KinovaBackend, SafetyWrapperBackend


class KinovaController:
    """Facade that exposes a stable, backend-agnostic Kinova API."""

    def __init__(self, backend: KinovaBackend, *, enforce_safety_wrapper: bool = True) -> None:
        if enforce_safety_wrapper and not isinstance(backend, SafetyWrapperBackend):
            backend = SafetyWrapperBackend(backend)
        self._backend = backend

    @classmethod
    def from_mode(cls, mode: str, **safety_kwargs: Any) -> "KinovaController":
        """Build a controller that wraps the selected backend in SafetyWrapperBackend."""
        from kinova_backend import make_kinova_api

        backend = make_kinova_api(mode, **safety_kwargs)
        return cls(backend)

    @property
    def dof(self) -> int:
        return self._backend.dof

    @property
    def arm_dof(self) -> int:
        return self._backend.arm_dof

    def init(self) -> None:
        self._backend.init()

    def close(self) -> None:
        self._backend.close()

    def move_home(self) -> None:
        self._backend.move_home()

    def reset_scene(self) -> None:
        """Reset the simulation scene."""
        self._backend.reset_scene()

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        self._backend.send_joint_position_rad(q_des)

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return end-effector pose as (position, quaternion)."""
        return self._backend.get_end_effector_pose()

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for target pose; returns joint targets in radians."""
        return self._backend.solve_ik(target_pos, target_quat, q_seed, move_wrist=move_wrist)

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for position only (no orientation constraint)."""
        return self._backend.solve_ik_position_only(target_pos, q_seed, move_wrist=move_wrist)

    def plan_to_pose(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        """Compute IK and command joints to reach target pose."""
        q_target = self.solve_ik(target_pos, target_quat, q_seed)
        self.send_joint_position_rad(q_target)
        return q_target

    def move_to_pose(self, pose: CartesianPose) -> list[float]:
        """Backward-compatible helper that plans from a CartesianPose."""
        return self.plan_to_pose(pose.position(), pose.quaternion())

    def step(self, **kwargs: Any) -> bool:
        return self._backend.step(**kwargs)

    def is_reached(self, **kwargs: Any) -> bool:
        return self._backend.is_reached(**kwargs)

    def get_joint_angles_rad(self) -> list[float]:
        return self._backend.get_joint_angles_rad()

    def get_joint_vel_rad(self) -> list[float]:
        return self._backend.get_joint_vel_rad()

    def get_target_joint_angles_rad(self) -> list[float]:
        """Read current target joint angles in radians."""
        return self._backend.get_target_joint_angles_rad()

    def set_gripper_percent(self, percent: float) -> None:
        """Set gripper opening percentage in [0.0, 1.0]."""
        self._backend.set_gripper_percent(percent)

    def get_finger_forces(self) -> dict:
        """Read current actuator forces for finger joints."""
        return self._backend.get_finger_forces()

    def open_fingers(self, percent: float) -> None:
        """Open the gripper by percent in [0.0, 1.0] (1.0 = fully open)."""
        self.set_gripper_percent(percent)

    def close_fingers(self, percent: float) -> None:
        """Close the gripper by percent in [0.0, 1.0] (1.0 = fully closed)."""
        self.set_gripper_percent(1.0 - float(percent))

    def rotate_wrist(self, angle_deg: float) -> None:
        """Rotate the wrist (last arm joint) by a relative angle in degrees."""
        self._backend.rotate_wrist(angle_deg)
