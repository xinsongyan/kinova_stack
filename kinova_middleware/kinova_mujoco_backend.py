from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import mujoco
import numpy as np

from kinova_backend import KinovaBackend

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from kinova_sim.controller import PDController  # noqa: E402
from kinova_sim.sim_env import SimEnv  # noqa: E402


DEFAULT_MODEL_PATH = os.path.join(
    _ROOT_DIR, "kinova_description", "mjcf", "m1n4s300_standalone.mjcf"
)

DEFAULT_JOINT_ORDER = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_finger_1",
    "joint_finger_tip_1",
    "joint_finger_2",
    "joint_finger_tip_2",
    "joint_finger_3",
    "joint_finger_tip_3",
)

DEFAULT_FINGER_JOINTS = (
    "joint_finger_1",
    "joint_finger_tip_1",
    "joint_finger_2",
    "joint_finger_tip_2",
    "joint_finger_3",
    "joint_finger_tip_3",
)

DEFAULT_EE_BODY = "link_4"
IK_MAX_ITERS = 500
IK_POS_TOL = 1e-3
IK_ROT_TOL = 1e-2
IK_DAMPING = 1e-2
IK_STEP_SIZE = 0.8
IK_ORIENTATION_WEIGHT = 0.3


class KinovaMuJoCoBackend(KinovaBackend):
    """MuJoCo backend that tracks joint targets with a PD control loop."""

    def __init__(
        self,
        model_path: str | None = None,
        joint_names: Sequence[str] | None = None,
        kp: Sequence[float] | float | None = None,
        kd: Sequence[float] | float | None = None,
        viewer: bool = True,
        initial_keyframe: str | None = "home",
        ee_body: str = DEFAULT_EE_BODY,
        ee_site: str | None = None,
        target_speed_rad_s: float | None = None,
    ) -> None:
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._joint_names = list(joint_names) if joint_names is not None else list(DEFAULT_JOINT_ORDER)
        self._kp = kp
        self._kd = kd
        self._viewer = viewer
        self._initial_keyframe = initial_keyframe
        self._ee_body_name = ee_body
        self._ee_site_name = ee_site
        self._target_speed_rad_s = self._resolve_target_speed(target_speed_rad_s)

        self._env: SimEnv | None = None
        self._controller: PDController | None = None
        self._joint_ids: list[int] = []
        self._qpos_adr: list[int] = []
        self._qvel_adr: list[int] = []
        self._actuator_ids: list[int] = []
        self._joint_limits: list[tuple[float, float]] = []
        self._finger_indices: list[int] = []
        self._arm_indices: list[int] = []
        self._arm_dof_adrs: list[int] = []
        self._q_target: np.ndarray | None = None
        self._q_target_desired: np.ndarray | None = None
        self._ee_body_id: int | None = None
        self._ee_site_id: int | None = None
        self._last_ik_iterations: int | None = None

    @property
    def dof(self) -> int:
        return len(self._joint_names)

    @property
    def arm_dof(self) -> int:
        return len(self._arm_indices)

    def init(self) -> None:
        if self._env is not None:
            return
        model_path = self._resolve_model_path(self._model_path)
        self._env = SimEnv(model_path, viewer=self._viewer)

        if self._initial_keyframe:
            self._env.set_model_keyframe(self._initial_keyframe)

        model = self._env.model
        data = self._env.data

        self._joint_ids = [self._require_joint_id(model, name) for name in self._joint_names]
        self._qpos_adr = [int(model.jnt_qposadr[jid]) for jid in self._joint_ids]
        self._qvel_adr = [int(model.jnt_dofadr[jid]) for jid in self._joint_ids]
        self._actuator_ids = [self._require_actuator_id(model, f"motor_{name}") for name in self._joint_names]
        if len(set(self._actuator_ids)) != len(self._actuator_ids):
            raise ValueError("Actuator mapping contains duplicates; check actuator names and ordering.")
        for actuator_id in self._actuator_ids:
            if actuator_id < 0 or actuator_id >= model.nu:
                raise ValueError(
                    f"Actuator id {actuator_id} out of range for nu={model.nu}; check model actuators."
                )

        self._joint_limits = []
        for jid in self._joint_ids:
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
            else:
                low, high = -np.inf, np.inf
            self._joint_limits.append((float(low), float(high)))

        self._finger_indices = [
            idx for idx, name in enumerate(self._joint_names) if name in DEFAULT_FINGER_JOINTS
        ]
        self._arm_indices = [
            idx for idx, name in enumerate(self._joint_names) if name not in DEFAULT_FINGER_JOINTS
        ]
        self._arm_dof_adrs = [self._qvel_adr[idx] for idx in self._arm_indices]

        self._ee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._ee_body_name
        )
        if self._ee_body_id < 0:
            raise ValueError(f"End-effector body '{self._ee_body_name}' not found in MuJoCo model.")
        self._ee_site_id = self._resolve_ee_site_id(model)

        qpos = data.qpos
        q_init = np.array([float(qpos[adr]) for adr in self._qpos_adr], dtype=float)
        self._q_target = q_init.copy()
        self._q_target_desired = q_init.copy()
        kp_default, kd_default = self._default_gains(self.dof)
        kp = self._resolve_gains(self._kp, kp_default)
        kd = self._resolve_gains(self._kd, kd_default)
        self._controller = PDController(kp, kd)

    def close(self) -> None:
        if self._env is None:
            return
        self._env.close()
        self._env = None
        self._controller = None
        self._q_target = None

    def move_home(self) -> None:
        env = self._require_env()
        target = None
        if self._initial_keyframe:
            keyframe = env.get_model_keyframe(self._initial_keyframe)
            if keyframe is not None:
                # Extract arm joint values using qpos addresses (robust to freejoints)
                kf_qpos = np.array(keyframe.qpos, dtype=float)
                all_qpos = np.array([kf_qpos[adr] for adr in self._qpos_adr], dtype=float)
                target = np.array([all_qpos[idx] for idx in self._arm_indices], dtype=float)
        if target is None:
            target = np.zeros(self.arm_dof, dtype=float)
        self.send_joint_position_rad(target)

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        self._require_env()
        n_arm = self.arm_dof
        if len(q_des) < n_arm:
            raise ValueError(f"Expected at least {n_arm} arm joint targets, got {len(q_des)}.")
        q_arm = np.array(q_des[: n_arm], dtype=float)
        # Preserve existing finger targets, only update arm joints
        q_full = self._require_q_target_desired().copy()
        for local_idx, joint_idx in enumerate(self._arm_indices):
            q_full[joint_idx] = q_arm[local_idx]
        self._q_target_desired = self._clip_q(q_full)
        if self._q_target is None:
            self._q_target = self._q_target_desired.copy()

    def step(
        self,
        pos_tol_rad: float = math.radians(0.8),
        vel_tol_rad_s: float = math.radians(2.0),
    ) -> bool:
        env = self._require_env()
        controller = self._require_controller()
        q_target_desired = self._require_q_target_desired()

        q = np.array([env.data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([env.data.qvel[adr] for adr in self._qvel_adr], dtype=float)
        if self._target_speed_rad_s is not None:
            q_target_filtered = self._require_q_target().copy()
            max_delta = float(self._target_speed_rad_s) * float(env.dt)
            delta = np.clip(q_target_desired - q_target_filtered, -max_delta, max_delta)
            q_target_filtered = q_target_filtered + delta
            self._q_target = q_target_filtered
            q_target = q_target_filtered
        else:
            q_target = q_target_desired
            self._q_target = q_target_desired

        tau = np.asarray(controller.compute(q, qd, q_target), dtype=float)
        if tau.shape != (len(self._actuator_ids),):
            raise ValueError(
                f"PD output must have shape ({len(self._actuator_ids)},), got {tau.shape}"
            )

        ctrl = np.zeros(env.model.nu, dtype=float)
        for tau_i, actuator_id in zip(tau, self._actuator_ids):
            tau_i = self._clip_actuator_force(env.model, actuator_id, float(tau_i))
            ctrl[actuator_id] = tau_i
        env.set_ctrl(ctrl)
        env.step()
        return self.is_desired_position_reached(pos_tol_rad=pos_tol_rad, vel_tol_rad_s=vel_tol_rad_s)

    def is_desired_position_reached(
        self,
        pos_tol_rad: float = math.radians(0.8),
        vel_tol_rad_s: float = math.radians(2.0),
    ) -> bool:
        env = self._require_env()
        q_target_desired = self._require_q_target_desired()
        q = np.array([env.data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([env.data.qvel[adr] for adr in self._qvel_adr], dtype=float)
        max_pos_err = float(np.max(np.abs(q_target_desired - q)))
        max_vel = float(np.max(np.abs(qd)))
        return bool(max_pos_err <= float(pos_tol_rad) and max_vel <= float(vel_tol_rad_s))

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
        env = self._require_env()
        q_target_desired = self._require_q_target_desired()
        q = np.array([env.data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([env.data.qvel[adr] for adr in self._qvel_adr], dtype=float)
        max_pos_err = float(np.max(np.abs(q_target_desired - q)))
        max_vel = float(np.max(np.abs(qd)))

        return bool(max_pos_err <= float(pos_tol_rad) and max_vel <= float(vel_tol_rad_s))

    def get_joint_angles_rad(self) -> list[float]:
        env = self._require_env()
        return [float(env.data.qpos[adr]) for adr in self._qpos_adr]

    def get_joint_vel_rad(self) -> list[float]:
        env = self._require_env()
        return [float(env.data.qvel[adr]) for adr in self._qvel_adr]

    def set_gripper_percent(self, percent: float) -> None:
        self._require_env()
        q_target = self._require_q_target_desired().copy()
        p = max(0.0, min(1.0, float(percent)))
        for idx in self._finger_indices:
            low, high = self._joint_limits[idx]
            if np.isfinite(low) and np.isfinite(high):
                q_target[idx] = low + (1.0 - p) * (high - low)
        self._q_target_desired = self._clip_q(q_target)
        if self._q_target is None:
            self._q_target = self._q_target_desired.copy()

    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        """Solve inverse kinematics for the requested Cartesian pose."""
        env = self._require_env()
        model = env.model
        data = env.data

        target_pos_arr = np.array(target_pos, dtype=float)
        if target_pos_arr.shape != (3,):
            raise ValueError("target_pos must have 3 elements.")

        target_quat_arr = np.array(target_quat, dtype=float)
        if target_quat_arr.shape != (4,):
            raise ValueError("target_quat must have 4 elements [qx, qy, qz, qw].")
        quat_norm = float(np.linalg.norm(target_quat_arr))
        if quat_norm < 1e-8:
            raise ValueError("target_quat must have non-zero norm.")
        target_quat_arr = target_quat_arr / quat_norm
        target_quat_wxyz = np.array(
            [target_quat_arr[3], target_quat_arr[0], target_quat_arr[1], target_quat_arr[2]],
            dtype=float,
        )

        qpos_initial = data.qpos.copy()
        qvel_initial = data.qvel.copy()

        if q_seed is not None:
            if len(q_seed) < self.dof:
                raise ValueError(f"Expected at least {self.dof} seed joints, got {len(q_seed)}.")
            q = np.array(q_seed[: self.dof], dtype=float)
        else:
            q = np.array([data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        q_arm = q[self._arm_indices].copy()

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        orientation_weight = IK_ORIENTATION_WEIGHT

        for it in range(IK_MAX_ITERS):
            data.qpos[:] = qpos_initial
            for idx, adr in enumerate(self._qpos_adr):
                data.qpos[adr] = q[idx]
            mujoco.mj_forward(model, data)

            pos, quat = self._get_ee_pose_wxyz(data)

            pos_err = target_pos_arr - pos
            rot_err = self._quat_error_vector(target_quat_wxyz, quat)

            if np.linalg.norm(pos_err) <= IK_POS_TOL and np.linalg.norm(rot_err) <= IK_ROT_TOL:
                self._last_ik_iterations = it + 1
                break

            self._fill_jacobian(model, data, jacp, jacr)
            cols = self._arm_dof_adrs
            jac = np.vstack((jacp[:, cols], jacr[:, cols]))

            error = np.concatenate((pos_err, rot_err))
            jac[:3, :] *= 1.0
            jac[3:, :] *= orientation_weight
            error[3:] *= orientation_weight

            jjt = jac @ jac.T
            damping = IK_DAMPING * IK_DAMPING * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt + damping, error)
            q_arm = q_arm + IK_STEP_SIZE * dq

            for local_idx, joint_idx in enumerate(self._arm_indices):
                low, high = self._joint_limits[joint_idx]
                if np.isfinite(low) or np.isfinite(high):
                    q_arm[local_idx] = float(np.clip(q_arm[local_idx], low, high))

            q[self._arm_indices] = q_arm
        else:
            self._last_ik_iterations = IK_MAX_ITERS

        data.qpos[:] = qpos_initial
        data.qvel[:] = qvel_initial
        mujoco.mj_forward(model, data)
        return [float(q[idx]) for idx in self._arm_indices]

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
    ) -> list[float]:
        """Solve IK for position only (no orientation constraint).

        Uses only the 3×N_arm positional Jacobian.  With a 4-DOF arm this
        gives 1 redundant DOF, allowing good reachability.
        """
        env = self._require_env()
        model = env.model
        data = env.data

        target_pos_arr = np.array(target_pos, dtype=float)
        if target_pos_arr.shape != (3,):
            raise ValueError("target_pos must have 3 elements.")

        qpos_initial = data.qpos.copy()
        qvel_initial = data.qvel.copy()

        if q_seed is not None:
            if len(q_seed) < self.dof:
                raise ValueError(f"Expected at least {self.dof} seed joints, got {len(q_seed)}.")
            q = np.array(q_seed[: self.dof], dtype=float)
        else:
            q = np.array([data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        q_arm = q[self._arm_indices].copy()

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)  # needed for API but unused

        for it in range(IK_MAX_ITERS):
            data.qpos[:] = qpos_initial
            for idx, adr in enumerate(self._qpos_adr):
                data.qpos[adr] = q[idx]
            mujoco.mj_forward(model, data)

            pos, _ = self._get_ee_pose_wxyz(data)
            pos_err = target_pos_arr - pos

            if np.linalg.norm(pos_err) <= IK_POS_TOL:
                self._last_ik_iterations = it + 1
                break

            self._fill_jacobian(model, data, jacp, jacr)
            cols = self._arm_dof_adrs
            jac_pos = jacp[:, cols]  # 3 × N_arm

            jjt = jac_pos @ jac_pos.T
            damping = IK_DAMPING * IK_DAMPING * np.eye(3)
            dq = jac_pos.T @ np.linalg.solve(jjt + damping, pos_err)
            q_arm = q_arm + IK_STEP_SIZE * dq

            for local_idx, joint_idx in enumerate(self._arm_indices):
                low, high = self._joint_limits[joint_idx]
                if np.isfinite(low) or np.isfinite(high):
                    q_arm[local_idx] = float(np.clip(q_arm[local_idx], low, high))

            q[self._arm_indices] = q_arm
        else:
            self._last_ik_iterations = IK_MAX_ITERS

        data.qpos[:] = qpos_initial
        data.qvel[:] = qvel_initial
        mujoco.mj_forward(model, data)
        return [float(q[idx]) for idx in self._arm_indices]

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        env = self._require_env()
        pos, quat_wxyz = self._get_ee_pose_wxyz(env.data)
        quat_xyzw = (float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))
        pos_xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
        return (pos_xyz, quat_xyzw)

    @property
    def last_ik_iterations(self) -> int | None:
        return self._last_ik_iterations

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(_ROOT_DIR, path))

    def _resolve_ee_site_id(self, model: mujoco.MjModel) -> int | None:
        candidates: list[str] = []
        if self._ee_site_name is not None:
            candidates.append(self._ee_site_name)
        else:
            candidates.extend(self._default_site_candidates())
            candidates.append(f"{self._ee_body_name}_site")
        for name in candidates:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                return site_id
        if self._ee_site_name is not None:
            raise ValueError(f"End-effector site '{self._ee_site_name}' not found in MuJoCo model.")
        return None

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

    @staticmethod
    def _require_joint_id(model: mujoco.MjModel, name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model.")
        return jid

    @staticmethod
    def _require_actuator_id(model: mujoco.MjModel, name: str) -> int:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"Actuator '{name}' not found in MuJoCo model.")
        return aid

    @staticmethod
    @staticmethod
    def _default_gains(dof: int) -> tuple[np.ndarray, np.ndarray]:
        if dof <= 4:
            # Arm only
            kp = np.full(dof, 800.0, dtype=float)
            kd = np.full(dof, 80.0, dtype=float)
            return kp, kd
        
        kp = np.full(dof, 1.0, dtype=float)
        kd = np.full(dof, 0.01, dtype=float)
        
        # Arm joints
        kp[:4] = 200.0
        kd[:4] = 8.0
        return kp, kd

    def _resolve_gains(
        self, values: Sequence[float] | float | None, default: np.ndarray
    ) -> np.ndarray:
        if values is None:
            return default
        arr = np.array(values, dtype=float)
        if arr.size == 1:
            return np.full(self.dof, float(arr.item()), dtype=float)
        if arr.size != self.dof:
            raise ValueError(f"Expected {self.dof} gain values, got {arr.size}.")
        return arr

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        q_clipped = q.copy()
        for idx, (low, high) in enumerate(self._joint_limits):
            if np.isfinite(low) or np.isfinite(high):
                q_clipped[idx] = float(np.clip(q_clipped[idx], low, high))
        return q_clipped

    @staticmethod
    def _clip_actuator_force(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
        if model.actuator_forcelimited[actuator_id]:
            low, high = model.actuator_forcerange[actuator_id]
            return float(np.clip(value, low, high))
        return value

    def _require_env(self) -> SimEnv:
        if self._env is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._env

    def _require_controller(self) -> PDController:
        if self._controller is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._controller

    def _require_q_target(self) -> np.ndarray:
        if self._q_target is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._q_target

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
