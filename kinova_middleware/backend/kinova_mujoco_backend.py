from __future__ import annotations

import math
import os
from typing import Any, Sequence

import mujoco
import numpy as np

from kinova_middleware.backend.config.kinova_gen3_lite import (
    DEFAULT_EE_BODY,
    DEFAULT_FINGER_JOINTS,
    DEFAULT_JOINT_ORDER,
    DEFAULT_KINOVA_MUJOCO_CONFIG,
    DEFAULT_MODEL_PATH,
)
from kinova_middleware.backend.config.robot_model import MuJoCoRobotConfig, _UNSET
from kinova_middleware.backend.kinova_backend import KinovaBackend
from kinova_middleware.backend.mujoco_ik import MuJoCoIKService, wrap_to_pi
from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter
from kinova_middleware.backend.services.gripper_service import MuJoCoGripperService
from kinova_middleware.backend.services.motion_service import MuJoCoArmControlService
from kinova_middleware.backend.services.object_query_service import MuJoCoObjectQueryService
from kinova_middleware.backend.services.scene_service import MuJoCoSceneService

class KinovaMuJoCoBackend(KinovaBackend):
    """MuJoCo backend with computed-torque control and jerk-limited trajectories."""

    def __init__(
        self,
        robot_config: MuJoCoRobotConfig | None = None,
        model_path: str | None = None,
        joint_names: Sequence[str] | None = None,
        kp: Sequence[float] | float | None = None,
        kd: Sequence[float] | float | None = None,
        viewer: bool = True,
        initial_keyframe: str | None | object = _UNSET,
        ee_body: str | object = _UNSET,
        ee_site: str | None | object = _UNSET,
        target_speed_rad_s: float | None = None,
    ) -> None:
        config = (robot_config or DEFAULT_KINOVA_MUJOCO_CONFIG).with_overrides(
            model_path=model_path,
            joint_names=joint_names,
            ee_body=ee_body,
            ee_site=ee_site,
            initial_keyframe=initial_keyframe,
        )

        self._config = config
        self._model_path = config.model_path
        self._joint_names = list(config.joint_names)
        self._kp = kp
        self._kd = kd
        self._viewer = viewer
        self._initial_keyframe = config.initial_keyframe
        self._ee_body_name = config.ee_body
        self._ee_site_name = config.ee_site
        self._target_speed_rad_s = self._resolve_target_speed(target_speed_rad_s)
        self._finger_joint_names = set(config.finger_joint_names)
        self._v_max_arm = np.array(config.v_max_arm, dtype=float)
        self._a_max_arm = np.array(config.a_max_arm, dtype=float)
        self._j_max_arm = np.array(config.j_max_arm, dtype=float)
        self._omega_arm = np.array(config.omega_arm, dtype=float)
        self._kp_arm = np.array(config.kp_arm, dtype=float)
        self._kd_arm = np.array(config.kd_arm, dtype=float)
        self._kp_finger = float(config.kp_finger)
        self._kd_finger = float(config.kd_finger)
        self._finger_bias_mode = config.finger_bias_mode
        self._control_dt = float(config.control_dt)
        self._runtime = MuJoCoRuntimeAdapter(
            model_path=config.model_path,
            joint_names=config.joint_names,
            ee_body_name=config.ee_body,
            ee_site_name=config.ee_site,
            viewer=self._viewer,
            site_candidates=self._default_site_candidates(),
        )

        self._finger_indices: list[int] = []
        self._arm_indices: list[int] = []
        self._arm_service: MuJoCoArmControlService | None = None
        self._gripper_service: MuJoCoGripperService | None = None
        self._ik_service: MuJoCoIKService | None = None
        self._scene_service: MuJoCoSceneService | None = None
        self._object_query_service: MuJoCoObjectQueryService | None = None
        self._q_target_desired: np.ndarray | None = None
        self._last_ik_iterations: int | None = None

    @property
    def dof(self) -> int:
        return len(self._joint_names)

    @property
    def arm_dof(self) -> int:
        return len(self._arm_indices)

    def init(self) -> None:
        if self._runtime.is_open():
            return
        self._runtime.open(initial_keyframe=self._initial_keyframe)
        model = self._runtime.model
        data = self._runtime.data

        self._finger_indices = [
            idx for idx, name in enumerate(self._joint_names) if name in self._finger_joint_names
        ]
        self._arm_indices = [
            idx for idx, name in enumerate(self._joint_names) if name not in self._finger_joint_names
        ]
        self._validate_arm_config_lengths(len(self._arm_indices))

        # ── Initial joint state ──────────────────────────────────────
        q_init = np.array([float(data.qpos[adr]) for adr in self._qpos_adr], dtype=float)
        self._q_target_desired = q_init.copy()
        self._arm_service = MuJoCoArmControlService(
            self._runtime,
            self._arm_indices,
            wrap_angle_fn=wrap_to_pi,
            v_max_arm=self._v_max_arm,
            a_max_arm=self._a_max_arm,
            j_max_arm=self._j_max_arm,
            omega_arm=self._omega_arm,
            kp_arm=self._kp_arm,
            kd_arm=self._kd_arm,
            control_dt=self._control_dt,
            j1_torque_limit=float(self._config.j1_torque_limit),
        )
        self._gripper_service = MuJoCoGripperService(
            self._runtime,
            self._finger_indices,
            kp_finger=self._kp_finger,
            kd_finger=self._kd_finger,
            finger_bias_mode=self._finger_bias_mode,
        )
        self._scene_service = MuJoCoSceneService(
            self._runtime,
            ee_pose_provider=self._get_ee_pose_wxyz,
        )
        self._object_query_service = MuJoCoObjectQueryService(self._runtime)

    def close(self) -> None:
        if not self._runtime.is_open():
            return
        self._runtime.close()
        self._arm_service = None
        self._gripper_service = None
        self._ik_service = None
        self._scene_service = None
        self._object_query_service = None

    def reset_scene(self) -> None:
        """Reset the simulation scene and controller state."""
        runtime = self._require_runtime()
        self._require_scene_service().reset(initial_keyframe=self._initial_keyframe)
            
        # Re-read initial state from data after reset
        q_init = np.array([float(runtime.data.qpos[adr]) for adr in self._qpos_adr], dtype=float)
        self._q_target_desired = q_init.copy()
        
        # Reset trajectory generator so the arm doesn't snap to the previous target
        self._require_arm_service().reset_from_joint_state(q_init)

    def move_home(self) -> None:
        runtime = self._require_runtime()
        target = None
        if self._initial_keyframe:
            keyframe = runtime.get_model_keyframe(self._initial_keyframe)
            if keyframe is not None:
                # Extract arm joint values using qpos addresses (robust to freejoints)
                kf_qpos = np.array(keyframe.qpos, dtype=float)
                all_qpos = np.array([kf_qpos[adr] for adr in self._qpos_adr], dtype=float)
                target = np.array([all_qpos[idx] for idx in self._arm_indices], dtype=float)
        if target is None:
            target = np.zeros(self.arm_dof, dtype=float)
        self.send_joint_position_rad(target)

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        self._require_runtime()
        q_curr = np.array(self.get_joint_angles_rad(), dtype=float)
        self._q_target_desired = self._require_arm_service().set_joint_target(
            q_des,
            q_curr,
            self._require_q_target_desired(),
        )

    def step(
        self,
        pos_tol_rad: float = math.radians(0.8),
        vel_tol_rad_s: float = math.radians(2.0),
    ) -> bool:
        runtime = self._require_runtime()
        model = runtime.model
        data = runtime.data
        ctrl = np.zeros(model.nu, dtype=float)
        for actuator_id, command in self._require_arm_service().compute_actuator_commands():
            ctrl[actuator_id] = command
        for actuator_id, command in self._require_gripper_service().compute_actuator_commands(
            self._require_q_target_desired()
        ):
            ctrl[actuator_id] = command

        # ══ h) Substeps (N × mj_step + 1 viewer sync) ──────────────
        data.ctrl[:] = ctrl
        runtime.step_n(self._require_arm_service().n_substeps)

        return self.is_desired_position_reached(pos_tol_rad=pos_tol_rad, vel_tol_rad_s=vel_tol_rad_s)

    def is_reached(self, **kwargs: Any) -> bool:
        return self.is_desired_position_reached(**kwargs)

    def _debug_counter(self) -> int:
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        return self._step_count

    def is_desired_position_reached(
        self,
        pos_tol_rad: float = math.radians(0.8),
        vel_tol_rad_s: float = math.radians(2.0),
    ) -> bool:
        return self._require_arm_service().is_reached(
            self._require_q_target_desired(),
            pos_tol_rad=pos_tol_rad,
            vel_tol_rad_s=vel_tol_rad_s,
        )

    def get_joint_angles_rad(self) -> list[float]:
        runtime = self._require_runtime()
        return [float(runtime.data.qpos[adr]) for adr in self._qpos_adr]

    def get_target_joint_angles_rad(self) -> list[float]:
        """Read current target joint angles (arm only) in radians."""
        q_target = self._require_q_target_desired()
        # Return only the arm joint components as a list
        return [float(q_target[i]) for i in self._arm_indices]

    def get_joint_vel_rad(self) -> list[float]:
        runtime = self._require_runtime()
        return [float(runtime.data.qvel[adr]) for adr in self._qvel_adr]

    def set_gripper_percent(self, percent: float) -> None:
        self._require_runtime()
        self._q_target_desired = self._require_gripper_service().set_target_percent(
            percent,
            self._require_q_target_desired(),
        )

    def get_gripper_state(self) -> dict:
        return self._require_gripper_service().get_gripper_state(self._require_q_target_desired())

    def wait_for_gripper(
        self,
        timeout_s: float = 5.0,
        hold_seconds: float = 0.2,
        hz: float = 500.0,
        pos_tol_rad: float = 0.05,
        vel_tol_rad_s: float = 0.2,
    ) -> bool:
        return self._require_gripper_service().wait_for_gripper(
            step_fn=lambda: self.step(),
            target_provider=self._require_q_target_desired,
            timeout_s=timeout_s,
            hold_seconds=hold_seconds,
            hz=hz,
            pos_tol_rad=pos_tol_rad,
            vel_tol_rad_s=vel_tol_rad_s,
        )

    def get_finger_forces(self) -> dict:
        return self._require_gripper_service().get_finger_forces()

    def get_object_pose(self, body_name: str) -> dict:
        """Read the Cartesian pose of a named body/site in the MuJoCo scene."""
        return self._require_object_query_service().get_object_pose(body_name)



    def _get_ik_service(self) -> MuJoCoIKService:
        if self._ik_service is None:
            self._ik_service = MuJoCoIKService(
                self._require_runtime(),
                self._arm_indices,
            )
        return self._ik_service

    def _require_arm_service(self) -> MuJoCoArmControlService:
        if self._arm_service is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._arm_service

    def _require_gripper_service(self) -> MuJoCoGripperService:
        if self._gripper_service is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._gripper_service

    def _require_scene_service(self) -> MuJoCoSceneService:
        if self._scene_service is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._scene_service

    def _require_object_query_service(self) -> MuJoCoObjectQueryService:
        if self._object_query_service is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._object_query_service

    def _validate_arm_config_lengths(self, arm_dof: int) -> None:
        arrays = {
            "v_max_arm": self._v_max_arm,
            "a_max_arm": self._a_max_arm,
            "j_max_arm": self._j_max_arm,
            "omega_arm": self._omega_arm,
            "kp_arm": self._kp_arm,
            "kd_arm": self._kd_arm,
        }
        for name, values in arrays.items():
            if len(values) != arm_dof:
                raise ValueError(
                    f"MuJoCo robot config field '{name}' has length {len(values)} but arm_dof is {arm_dof}."
                )

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve inverse kinematics for the requested Cartesian pose.
        
        If move_wrist=False, the requested quaternion is advisory/ignored 
        and the solve purely targets position-only to preserve the wrist joint.
        """
        if not move_wrist:
            return self.solve_ik_position_only(target_pos, q_seed, move_wrist=False)

        target_quat_arr = np.array(target_quat, dtype=float)
        quat_norm = float(np.linalg.norm(target_quat_arr))
        if quat_norm > 1e-8:
            target_quat_arr = target_quat_arr / quat_norm
        else:
            target_quat_arr = np.array([1.0, 0.0, 0.0, 0.0])
            
        # Convert quat (xyzw) to rotation matrix to extract the target X-axis (direction out of gripper)
        from scipy.spatial.transform import Rotation
        try:
            r = Rotation.from_quat(target_quat_arr)
            # MuJoCo is wxyz, but standard scipy is xyzw. Since the user might pass wxyz or xyzw, let's just 
            # use our custom _quat_multiply or assume it's wxyz based on _get_ee_pose_wxyz.
            pass
        except:
            pass
            
        # For simplicity, if they call solve_ik, we can map it to our 5-DOF IK by extracting 
        # the X-axis from the quaternion. But since the previous API expects WXYZ or XYZW, 
        # let's just explicitly compute the rotation matrix.
        
        qw, qx, qy, qz = target_quat_arr if len(target_quat_arr)==4 else (1,0,0,0)
        # Assuming wxyz (based on existing backend code `target_quat_wxyz = np.array([target_quat_arr[3], target_quat_arr[0]...`)
        # Wait, existing code: target_quat_arr[3] is w, so it's xyzw.
        qx, qy, qz, qw = target_quat_arr
        
        # rotation matrix first column (x-axis)
        target_x = np.array([
            1 - 2*qy**2 - 2*qz**2,
            2*qx*qy + 2*qz*qw,
            2*qx*qz - 2*qy*qw
        ])
        
        active_dof = self.arm_dof if move_wrist else self.arm_dof - 1
        ik_service = self._get_ik_service()
        best_q = ik_service.solve(np.array(target_pos), target_x, q_seed, active_dof=active_dof)
        self._last_ik_iterations = ik_service.last_iterations
        return [float(x) for x in best_q]

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for position only (no orientation constraint)."""
        active_dof = self.arm_dof if move_wrist else self.arm_dof - 1
        ik_service = self._get_ik_service()
        best_q = ik_service.solve(np.array(target_pos), None, q_seed, active_dof=active_dof)
        self._last_ik_iterations = ik_service.last_iterations
        return [float(x) for x in best_q]

    def solve_ik_z_down(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for position accuracy while pointing straight down."""
        active_dof = self.arm_dof if move_wrist else self.arm_dof - 1
        ik_service = self._get_ik_service()
        # Downward vector is -Z in world coords
        best_q = ik_service.solve(
            np.array(target_pos),
            np.array([0.0, 0.0, -1.0]),
            q_seed,
            active_dof=active_dof,
        )
        self._last_ik_iterations = ik_service.last_iterations
        return [float(x) for x in best_q]


    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return self._require_scene_service().get_end_effector_pose()

    @property
    def last_ik_iterations(self) -> int | None:
        return self._last_ik_iterations

    def _get_ee_pose_wxyz(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        if self._ee_site_id is not None:
            pos = np.array(data.site_xpos[self._ee_site_id], dtype=float)
            xmat = np.array(data.site_xmat[self._ee_site_id], dtype=float).reshape(3, 3)
            quat = self._quat_wxyz_from_mat(xmat)
            return pos, quat
        body_id = self._require_ee_body()
        pos = np.array(data.xpos[body_id], dtype=float)
        quat = np.array(data.xquat[body_id], dtype=float)
        return pos, quat

    def _fill_jacobian(
        self, model: mujoco.MjModel, data: mujoco.MjData, jacp: np.ndarray, jacr: np.ndarray
    ) -> None:
        if self._ee_site_id is not None:
            mujoco.mj_jacSite(model, data, jacp, jacr, self._ee_site_id)
        else:
            ee_body = self._require_ee_body()
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_body)

    def _require_runtime(self) -> MuJoCoRuntimeAdapter:
        if not self._runtime.is_open():
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._runtime

    @property
    def _joint_ids(self) -> tuple[int, ...]:
        return self._require_runtime().joint_ids

    @property
    def _qpos_adr(self) -> tuple[int, ...]:
        return self._require_runtime().qpos_adr

    @property
    def _qvel_adr(self) -> tuple[int, ...]:
        return self._require_runtime().qvel_adr

    @property
    def _actuator_ids(self) -> tuple[int, ...]:
        return self._require_runtime().actuator_ids

    @property
    def _joint_limits(self) -> tuple[tuple[float, float], ...]:
        return self._require_runtime().joint_limits

    @property
    def _continuous_indices(self) -> tuple[int, ...]:
        return self._require_runtime().continuous_indices

    @property
    def _ee_body_id(self) -> int | None:
        return self._require_runtime().ee_body_id

    @property
    def _ee_site_id(self) -> int | None:
        return self._require_runtime().ee_site_id

    def _require_q_target_desired(self) -> np.ndarray:
        if self._q_target_desired is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._q_target_desired

    def _require_ee_body(self) -> int:
        if self._ee_body_id is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._ee_body_id

    @staticmethod
    def _default_site_candidates() -> list[str]:
        return ["tool_tip", "ee_site", "end_effector"]

    @staticmethod
    def _quat_wxyz_from_mat(mat: np.ndarray) -> np.ndarray:
        m = np.array(mat, dtype=float).reshape(3, 3)
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz], dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm > 1e-8:
            quat = quat / norm
        return quat

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    def _quat_error_vector(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        q_err = self._quat_multiply(target, self._quat_conjugate(current))
        if q_err[0] < 0:
            q_err = -q_err
        q_err[0] = np.clip(q_err[0], -1.0, 1.0)
        angle = 2.0 * math.acos(q_err[0])
        if angle < 1e-8:
            return 2.0 * q_err[1:]
        axis = q_err[1:] / math.sin(angle / 2.0)
        return axis * angle

    @staticmethod
    def _resolve_target_speed(value: float | None) -> float | None:
        if value is not None:
            speed = float(value)
            if speed <= 0:
                raise ValueError("target_speed_rad_s must be positive.")
            return speed
        env_val = os.getenv("KINOVA_TARGET_SPEED_RAD_S")
        if env_val is None or env_val == "":
            return None
        speed = float(env_val)
        if speed <= 0:
            raise ValueError("KINOVA_TARGET_SPEED_RAD_S must be positive.")
        return speed
