from __future__ import annotations

import math
import os
import sys
import time
from typing import Sequence

from kinova_backend import KinovaBackend

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KINOVA_API_PY_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "kinova-api-python"))
if _KINOVA_API_PY_DIR not in sys.path:
    sys.path.append(_KINOVA_API_PY_DIR)

from kinova_api import (  # noqa: E402
    ANGULAR_POSITION,
    HAND_NOMOVEMENT,
    KinovaAPI,
    TrajectoryPoint,
)


class KinovaSDKBackend(KinovaBackend):
    """Backend that talks to the real Kinova arm via the USB SDK (ctypes)."""

    def __init__(
        self,
        lib_dir: str | None = None,
        device_index: int = 0,
        ik_model_path: str | None = None,
        ik_joint_names: Sequence[str] | None = None,
        ik_ee_body: str | None = None,
        ik_ee_site: str | None = None,
    ) -> None:
        self._lib_dir = lib_dir or os.getenv("KINOVA_API_DIR")
        self._device_index = device_index
        self._ik_model_path = ik_model_path
        self._ik_joint_names = list(ik_joint_names) if ik_joint_names is not None else None
        self._ik_ee_body = ik_ee_body
        self._ik_ee_site = ik_ee_site
        self._ik_backend = None
        self._api: KinovaAPI | None = None
        self._last_angles: list[float] | None = None
        self._last_time: float | None = None

    @property
    def dof(self) -> int:
        return 7

    @property
    def arm_dof(self) -> int:
        return 7

    def init(self) -> None:
        if self._api is not None:
            return
        api = KinovaAPI(lib_dir=self._lib_dir)
        api.init()
        devices = api.list_devices()
        if not devices:
            raise RuntimeError("No Kinova devices detected. Check USB connection and permissions.")
        if not (0 <= self._device_index < len(devices)):
            raise IndexError(f"Device index {self._device_index} out of range for {len(devices)} device(s).")
        api.set_active_device(devices[self._device_index])
        api.init_fingers()
        self._api = api
        self._last_angles = None
        self._last_time = None

    def close(self) -> None:
        if self._api is not None:
            self._api.close()
            self._api = None
        if self._ik_backend is not None:
            self._ik_backend.close()
            self._ik_backend = None

    def move_home(self) -> None:
        api = self._require_api()
        api.move_home()

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        api = self._require_api()
        n_arm = self.arm_dof
        if len(q_des) < n_arm:
            raise ValueError(f"Expected at least {n_arm} arm joint targets, got {len(q_des)}.")

        deg = [math.degrees(float(q)) for q in q_des[: n_arm]]
        point = TrajectoryPoint()
        point.Position.Type = ANGULAR_POSITION
        point.Position.HandMode = HAND_NOMOVEMENT
        point.Position.Actuators.Actuator1 = deg[0]
        point.Position.Actuators.Actuator2 = deg[1]
        point.Position.Actuators.Actuator3 = deg[2]
        point.Position.Actuators.Actuator4 = deg[3]
        point.Position.Actuators.Actuator5 = deg[4]
        point.Position.Actuators.Actuator6 = deg[5]
        point.Position.Actuators.Actuator7 = deg[6]
        point.LimitationsActive = 0
        point.SynchroType = 0
        rc = api._usb.SendBasicTrajectory(point)
        api._ensure_ok(rc, "SendBasicTrajectory(angular_position)")

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        api = self._require_api()
        pos, quat = api.get_end_effector_pos_quat()
        pos_out = (float(pos[0]), float(pos[1]), float(pos[2]))
        quat_out = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return (pos_out, quat_out)

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        if q_seed is None:
            q_seed = self.get_joint_angles_rad()
        ik_backend = self._ensure_ik_backend()
        if ik_backend.dof != self.dof:
            raise ValueError(
                f"IK model DOF ({ik_backend.dof}) does not match hardware DOF ({self.dof}). "
                "Provide a matching ik_model_path and ik_joint_names."
            )
        return ik_backend.solve_ik(target_pos, target_quat, q_seed)

    def step(self) -> bool:
        # Real hardware executes commands asynchronously; no explicit step required.
        return False

    def get_joint_angles_rad(self) -> list[float]:
        api = self._require_api()
        return api.get_joint_angles_rad()

    def get_joint_vel_rad(self) -> list[float]:
        now = time.monotonic()
        angles = self.get_joint_angles_rad()
        if self._last_angles is None or self._last_time is None:
            self._last_angles = angles
            self._last_time = now
            return [0.0] * len(angles)
        dt = now - self._last_time
        if dt <= 1e-6:
            return [0.0] * len(angles)
        vel = [(a - b) / dt for a, b in zip(angles, self._last_angles)]
        self._last_angles = angles
        self._last_time = now
        return vel

    def set_gripper_percent(self, percent: float) -> None:
        api = self._require_api()
        p = max(0.0, min(1.0, float(percent)))
        api.set_fingers_percent(p, p, p)

    @staticmethod
    def quat_to_euler_xyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        """Convert quaternion (qx, qy, qz, qw) to XYZ intrinsic Euler angles (rad)."""
        t0 = 2.0 * (qw * qx + qy * qz)
        t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
        theta_x = math.atan2(t0, t1)

        t2 = 2.0 * (qw * qy - qz * qx)
        t2 = max(-1.0, min(1.0, t2))
        theta_y = math.asin(t2)

        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        theta_z = math.atan2(t3, t4)

        return (theta_x, theta_y, theta_z)

    def _require_api(self) -> KinovaAPI:
        if self._api is None:
            raise RuntimeError("KinovaSDKBackend.init() must be called before use.")
        return self._api

    def _ensure_ik_backend(self):
        if self._ik_backend is None:
            from kinova_mujoco_backend import KinovaMuJoCoBackend

            kwargs: dict[str, object] = {"viewer": False}
            if self._ik_model_path is not None:
                kwargs["model_path"] = self._ik_model_path
            if self._ik_joint_names is not None:
                kwargs["joint_names"] = self._ik_joint_names
            if self._ik_ee_body is not None:
                kwargs["ee_body"] = self._ik_ee_body
            if self._ik_ee_site is not None:
                kwargs["ee_site"] = self._ik_ee_site
            self._ik_backend = KinovaMuJoCoBackend(**kwargs)
            self._ik_backend.init()
        return self._ik_backend
