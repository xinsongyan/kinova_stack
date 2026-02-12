from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence


_QUAT_NORM_EPS = 1e-8
_QUAT_NORM_TOL = 1e-3


@dataclass(slots=True)
class CartesianPose:
    """Cartesian end-effector pose (meters + unit quaternion)."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Cartesian pose values must be finite.")
        norm = math.sqrt(self.qx * self.qx + self.qy * self.qy + self.qz * self.qz + self.qw * self.qw)
        if norm < _QUAT_NORM_EPS:
            raise ValueError("Quaternion norm is too small to normalize.")
        if abs(norm - 1.0) > _QUAT_NORM_TOL:
            self.qx /= norm
            self.qy /= norm
            self.qz /= norm
            self.qw /= norm
        else:
            # Normalize anyway to keep consistency across callers.
            self.qx /= norm
            self.qy /= norm
            self.qz /= norm
            self.qw /= norm

    @classmethod
    def from_position_orientation(
        cls, position: Sequence[float], orientation: Sequence[float]
    ) -> "CartesianPose":
        if len(position) != 3:
            raise ValueError("Position must have 3 elements [x, y, z].")
        if len(orientation) != 4:
            raise ValueError("Orientation must have 4 elements [qx, qy, qz, qw].")
        return cls(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        )

    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def quaternion(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)

    def with_position(self, x: float, y: float, z: float) -> "CartesianPose":
        return CartesianPose(float(x), float(y), float(z), self.qx, self.qy, self.qz, self.qw)

    def is_unit_quaternion(self, tol: float = _QUAT_NORM_TOL) -> bool:
        norm = math.sqrt(self.qx * self.qx + self.qy * self.qy + self.qz * self.qz + self.qw * self.qw)
        return abs(norm - 1.0) <= tol


class KinovaBackend(ABC):
    """Abstract interface for Kinova execution backends (sim or real)."""

    @property
    @abstractmethod
    def dof(self) -> int:
        """Total number of joints including fingers (constant for the backend)."""

    @property
    @abstractmethod
    def arm_dof(self) -> int:
        """Number of arm-only joints (excluding fingers)."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the backend (SDK connection or simulation setup)."""

    @abstractmethod
    def close(self) -> None:
        """Clean shutdown of backend resources."""

    @abstractmethod
    def move_home(self) -> None:
        """Send the robot to its home configuration."""

    @abstractmethod
    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        """Command arm joint positions in radians (arm joints only, not fingers)."""

    @abstractmethod
    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return end-effector pose as (position, quaternion)."""

    @abstractmethod
    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        """Solve IK for target pose; returns joint targets in radians."""

    @abstractmethod
    def step(self) -> bool:
        """Advance backend control loop and report whether current target is reached."""

    @abstractmethod
    def get_joint_angles_rad(self) -> list[float]:
        """Read current joint angles in radians."""

    @abstractmethod
    def get_joint_vel_rad(self) -> list[float]:
        """Read current joint velocities in radians/sec (or estimated)."""

    @abstractmethod
    def set_gripper_percent(self, percent: float) -> None:
        """Control gripper opening as a percentage in [0.0, 1.0]."""


class SafetyWrapperBackend(KinovaBackend):
    """Backend wrapper that enforces joint/velocity/workspace safety constraints."""

    def __init__(
        self,
        inner: KinovaBackend,
        joint_limits: Sequence[tuple[float, float]] | None = None,
        max_joint_step: Sequence[float] | float | None = None,
        max_joint_velocity: Sequence[float] | float | None = None,
        workspace_validator: Callable[[Sequence[float]], bool] | None = None,
        workspace_clipper: Callable[[Sequence[float]], Sequence[float]] | None = None,
        cartesian_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        cartesian_no_go_boxes: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
        cartesian_no_go_spheres: Sequence[tuple[Sequence[float], float]] | None = None,
        cartesian_validator: Callable[[CartesianPose], bool] | None = None,
        on_violation: str = "clip",
        command_period_s: float | None = None,
    ) -> None:
        self._inner = inner
        self._joint_limits = self._normalize_joint_limits(joint_limits)
        self._max_joint_step = self._normalize_limit(max_joint_step, "max_joint_step")
        self._max_joint_velocity = self._normalize_limit(
            max_joint_velocity, "max_joint_velocity"
        )
        self._workspace_validator = workspace_validator
        self._workspace_clipper = workspace_clipper
        self._cartesian_bounds = self._normalize_cartesian_bounds(cartesian_bounds)
        self._cartesian_no_go_boxes = self._normalize_cartesian_boxes(cartesian_no_go_boxes)
        self._cartesian_no_go_spheres = self._normalize_cartesian_spheres(cartesian_no_go_spheres)
        self._cartesian_validator = cartesian_validator
        if on_violation not in ("clip", "reject"):
            raise ValueError("on_violation must be 'clip' or 'reject'.")
        self._on_violation = on_violation
        if command_period_s is not None and command_period_s <= 0:
            raise ValueError("command_period_s must be positive when provided.")
        self._command_period_s = command_period_s
        self._last_command_time: float | None = None

    @property
    def dof(self) -> int:
        return self._inner.dof

    @property
    def arm_dof(self) -> int:
        return self._inner.arm_dof

    def init(self) -> None:
        self._inner.init()

    def close(self) -> None:
        self._inner.close()

    def move_home(self) -> None:
        self._inner.move_home()

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        n_arm = self.arm_dof
        if len(q_des) < n_arm:
            raise ValueError(f"Expected at least {n_arm} arm joint targets, got {len(q_des)}.")
        q_target = [float(q) for q in q_des[: n_arm]]
        if any(not math.isfinite(q) for q in q_target):
            raise ValueError("Joint targets must be finite.")

        q_current_all = self._inner.get_joint_angles_rad()
        q_current = [float(q) for q in q_current_all[: n_arm]]

        q_target = self._apply_joint_limits(q_target)
        q_target = self._apply_step_limits(q_current, q_target)
        q_target = self._apply_velocity_limits(q_current, q_target)
        q_target = self._apply_workspace_limits(q_target)
        q_target = self._apply_joint_limits(q_target)

        self._inner.send_joint_position_rad(q_target)
        self._last_command_time = time.monotonic()

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return self._inner.get_end_effector_pose()

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        safe_pose = CartesianPose.from_position_orientation(target_pos, target_quat)
        safe_pose = self._apply_cartesian_limits(safe_pose)

        try:
            q_candidate = self._inner.solve_ik(
                safe_pose.position(), safe_pose.quaternion(), q_seed
            )
        except Exception as exc:
            raise ValueError("Cartesian pose rejected: IK failed.") from exc

        if self._joint_limits is not None:
            for idx, (low, high) in enumerate(self._joint_limits):
                value = float(q_candidate[idx])
                if value < low or value > high:
                    raise ValueError("Cartesian pose rejected: IK solution violates joint limits.")

        return [float(q) for q in q_candidate[: self.arm_dof]]

    def step(self) -> bool:
        return self._inner.step()

    def get_joint_angles_rad(self) -> list[float]:
        return self._inner.get_joint_angles_rad()

    def get_joint_vel_rad(self) -> list[float]:
        return self._inner.get_joint_vel_rad()

    def set_gripper_percent(self, percent: float) -> None:
        self._inner.set_gripper_percent(percent)

    def _normalize_joint_limits(
        self, joint_limits: Sequence[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        if joint_limits is None:
            return None
        limits = list(joint_limits)
        if len(limits) != self.dof:
            raise ValueError(f"Expected {self.dof} joint limits, got {len(limits)}.")
        normalized: list[tuple[float, float]] = []
        for low, high in limits:
            low_f = float(low)
            high_f = float(high)
            if low_f > high_f:
                raise ValueError("Joint limit min must be <= max.")
            normalized.append((low_f, high_f))
        return normalized

    def _normalize_limit(
        self, limit: Sequence[float] | float | None, name: str
    ) -> list[float] | None:
        if limit is None:
            return None
        if isinstance(limit, (int, float)):
            value = float(limit)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            return [value] * self.dof
        values = [float(v) for v in limit]
        if len(values) != self.dof:
            raise ValueError(f"Expected {self.dof} {name} values, got {len(values)}.")
        if any(v <= 0 for v in values):
            raise ValueError(f"{name} values must be positive.")
        return values

    def _apply_joint_limits(self, q_target: list[float]) -> list[float]:
        if self._joint_limits is None:
            return q_target
        clipped = q_target.copy()
        violated = False
        for idx, (low, high) in enumerate(self._joint_limits):
            value = clipped[idx]
            if value < low or value > high:
                violated = True
                if self._on_violation == "clip":
                    clipped[idx] = max(low, min(high, value))
        if violated and self._on_violation == "reject":
            raise ValueError("Joint limits violated.")
        return clipped

    def _apply_step_limits(self, q_current: list[float], q_target: list[float]) -> list[float]:
        if self._max_joint_step is None:
            return q_target
        clipped = q_target.copy()
        violated = False
        for idx, max_step in enumerate(self._max_joint_step):
            delta = clipped[idx] - q_current[idx]
            if abs(delta) > max_step:
                violated = True
                if self._on_violation == "clip":
                    clipped[idx] = q_current[idx] + math.copysign(max_step, delta)
        if violated and self._on_violation == "reject":
            raise ValueError("Maximum joint step exceeded.")
        return clipped

    def _apply_velocity_limits(self, q_current: list[float], q_target: list[float]) -> list[float]:
        if self._max_joint_velocity is None:
            return q_target
        dt = self._command_period_s
        if dt is None and self._last_command_time is not None:
            dt = time.monotonic() - self._last_command_time
        if dt is None or dt <= 0:
            return q_target
        clipped = q_target.copy()
        violated = False
        for idx, max_vel in enumerate(self._max_joint_velocity):
            delta = clipped[idx] - q_current[idx]
            if abs(delta) / dt > max_vel:
                violated = True
                if self._on_violation == "clip":
                    clipped[idx] = q_current[idx] + math.copysign(max_vel * dt, delta)
        if violated and self._on_violation == "reject":
            raise ValueError("Maximum joint velocity exceeded.")
        return clipped

    def _apply_workspace_limits(self, q_target: list[float]) -> list[float]:
        if self._workspace_validator is None:
            return q_target
        if self._workspace_validator(q_target):
            return q_target
        if self._on_violation == "clip" and self._workspace_clipper is not None:
            clipped = [float(q) for q in self._workspace_clipper(q_target)]
            if len(clipped) != self.dof:
                raise ValueError("Workspace clipper returned the wrong number of joints.")
            if self._workspace_validator(clipped):
                return clipped
        raise ValueError("Workspace constraints violated.")

    def _normalize_cartesian_bounds(
        self, bounds: tuple[Sequence[float], Sequence[float]] | None
    ) -> tuple[list[float], list[float]] | None:
        if bounds is None:
            return None
        low = self._normalize_xyz(bounds[0], "cartesian_bounds min")
        high = self._normalize_xyz(bounds[1], "cartesian_bounds max")
        for idx in range(3):
            if low[idx] > high[idx]:
                raise ValueError("Cartesian bounds min must be <= max.")
        return (low, high)

    def _normalize_cartesian_boxes(
        self, boxes: Sequence[tuple[Sequence[float], Sequence[float]]] | None
    ) -> list[tuple[list[float], list[float]]]:
        if boxes is None:
            return []
        normalized: list[tuple[list[float], list[float]]] = []
        for low, high in boxes:
            low_xyz = self._normalize_xyz(low, "cartesian_no_go_box min")
            high_xyz = self._normalize_xyz(high, "cartesian_no_go_box max")
            for idx in range(3):
                if low_xyz[idx] > high_xyz[idx]:
                    raise ValueError("No-go box min must be <= max.")
            normalized.append((low_xyz, high_xyz))
        return normalized

    def _normalize_cartesian_spheres(
        self, spheres: Sequence[tuple[Sequence[float], float]] | None
    ) -> list[tuple[list[float], float]]:
        if spheres is None:
            return []
        normalized: list[tuple[list[float], float]] = []
        for center, radius in spheres:
            center_xyz = self._normalize_xyz(center, "cartesian_no_go_sphere center")
            r = float(radius)
            if r <= 0:
                raise ValueError("No-go sphere radius must be positive.")
            normalized.append((center_xyz, r))
        return normalized

    def _normalize_xyz(self, value: Sequence[float], name: str) -> list[float]:
        values = [float(v) for v in value]
        if len(values) != 3:
            raise ValueError(f"{name} must have 3 elements.")
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"{name} values must be finite.")
        return values

    def _apply_cartesian_limits(self, pose: CartesianPose) -> CartesianPose:
        if not pose.is_unit_quaternion():
            raise ValueError("Cartesian pose quaternion must be unit length.")
        if self._cartesian_validator is not None and not self._cartesian_validator(pose):
            raise ValueError("Cartesian pose rejected by custom validator.")

        out_pose = pose
        if self._cartesian_bounds is not None:
            low, high = self._cartesian_bounds
            clipped = [
                max(low[0], min(high[0], pose.x)),
                max(low[1], min(high[1], pose.y)),
                max(low[2], min(high[2], pose.z)),
            ]
            if (clipped[0], clipped[1], clipped[2]) != (pose.x, pose.y, pose.z):
                if self._on_violation == "reject":
                    raise ValueError("Cartesian bounds violated.")
                out_pose = pose.with_position(*clipped)

        for low, high in self._cartesian_no_go_boxes:
            if (
                low[0] <= out_pose.x <= high[0]
                and low[1] <= out_pose.y <= high[1]
                and low[2] <= out_pose.z <= high[2]
            ):
                raise ValueError("Cartesian pose rejected: inside no-go box.")

        for center, radius in self._cartesian_no_go_spheres:
            dx = out_pose.x - center[0]
            dy = out_pose.y - center[1]
            dz = out_pose.z - center[2]
            if dx * dx + dy * dy + dz * dz <= radius * radius:
                raise ValueError("Cartesian pose rejected: inside no-go sphere.")

        return out_pose


def make_kinova_api(mode: str, **safety_kwargs: Any) -> KinovaBackend:
    """Factory for Kinova backends with safety wrapper."""
    mode_key = mode.strip().lower()
    if mode_key == "sim":
        from kinova_mujoco_backend import KinovaMuJoCoBackend

        inner = KinovaMuJoCoBackend()
    elif mode_key == "real":
        from kinova_sdk_backend import KinovaSDKBackend

        inner = KinovaSDKBackend()
    else:
        raise ValueError("mode must be 'sim' or 'real'.")
    return SafetyWrapperBackend(inner, **safety_kwargs)
