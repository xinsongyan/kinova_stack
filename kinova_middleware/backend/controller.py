from __future__ import annotations

from typing import Any, Sequence

from kinova_middleware.backend.interfaces.capabilities import (
    BackendCapability,
    CapabilityLike,
)
from kinova_middleware.backend.kinova_backend import (
    CartesianPose,
    KinovaBackend,
    SafetyWrapperBackend,
)


class KinovaController:
    """Facade that exposes a stable, backend-agnostic Kinova API."""

    def __init__(self, backend: KinovaBackend, *, enforce_safety_wrapper: bool = True) -> None:
        if enforce_safety_wrapper and not isinstance(backend, SafetyWrapperBackend):
            backend = SafetyWrapperBackend(backend)
        self._backend = backend

    @classmethod
    def from_mode(cls, mode: str, **backend_kwargs: Any) -> "KinovaController":
        from kinova_middleware.backend.factory import build_backend

        return cls(build_backend(mode, **backend_kwargs))

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
        self._backend.reset_scene()

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        self._backend.send_joint_position_rad(q_des)

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return self._backend.get_end_effector_pose()

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        return self._backend.solve_ik(target_pos, target_quat, q_seed, move_wrist=move_wrist)

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        return self._backend.solve_ik_position_only(target_pos, q_seed, move_wrist=move_wrist)

    def plan_to_pose(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        q_target = self.solve_ik(target_pos, target_quat, q_seed)
        self.send_joint_position_rad(q_target)
        return q_target

    def move_to_pose(self, pose: CartesianPose) -> list[float]:
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
        return self._backend.get_target_joint_angles_rad()

    def set_gripper_percent(self, percent: float) -> None:
        self._backend.set_gripper_percent(percent)

    def get_finger_forces(self) -> dict:
        return self._backend.get_finger_forces()

    def get_gripper_state(self) -> dict:
        return self._backend.get_gripper_state()

    def supported_capabilities(self) -> frozenset[BackendCapability]:
        return self._backend.supported_capabilities()

    def supports_capability(self, capability: CapabilityLike) -> bool:
        normalized = capability if isinstance(capability, BackendCapability) else BackendCapability(capability)
        return normalized in self.supported_capabilities()

    def get_object_pose(self, body_name: str) -> dict:
        return self._backend.get_object_pose(body_name)

    def wait_for_gripper(
        self,
        timeout_s: float = 5.0,
        hold_seconds: float = 0.2,
        hz: float = 500.0,
        pos_tol_rad: float = 0.05,
        vel_tol_rad_s: float = 0.2,
    ) -> bool:
        return self._backend.wait_for_gripper(
            timeout_s=timeout_s,
            hold_seconds=hold_seconds,
            hz=hz,
            pos_tol_rad=pos_tol_rad,
            vel_tol_rad_s=vel_tol_rad_s,
        )

    def open_fingers(self, percent: float) -> None:
        self.set_gripper_percent(percent)

    def close_fingers(self, percent: float) -> None:
        self.set_gripper_percent(1.0 - float(percent))

    def rotate_wrist(self, angle_deg: float) -> None:
        self._backend.rotate_wrist(angle_deg)
