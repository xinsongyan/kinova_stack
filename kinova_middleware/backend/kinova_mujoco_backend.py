from __future__ import annotations

import math
import os
import sys
from typing import Any, Sequence

import mujoco
import numpy as np

from kinova_backend import KinovaBackend

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from kinova_sim.controller import ComputedTorqueController, PDController  # noqa: E402
from kinova_sim.governor import ReferenceGovernor  # noqa: E402
from kinova_sim.sim_env import SimEnv  # noqa: E402
from kinova_sim.trajectory import TrajectoryGenerator  # noqa: E402


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
IK_ORIENTATION_WEIGHT = 0.15

def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def joint_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    continuous_indices: list[int],
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute shortest-path joint distance between two configurations.

    Continuous joints are wrapped to [-pi, pi] before distance calculation.

    Args:
        q1: first joint configuration
        q2: second joint configuration
        continuous_indices: indices of continuous joints
        weights: optional weighting vector

    Returns:
        Euclidean joint distance
    """

    diff = q1 - q2

    # Wrap continuous joints
    for idx in continuous_indices:
        diff[idx] = wrap_to_pi(diff[idx])

    # Apply weights AFTER wrapping
    if weights is not None:
        diff = diff * weights

    return float(np.linalg.norm(diff))

def shortest_joint_configuration(q_current, q_solution, continuous_indices):
    q_fixed = q_solution.copy()

    for idx in continuous_indices:
        delta = q_solution[idx] - q_current[idx]
        delta_wrapped = wrap_to_pi(delta)
        q_fixed[idx] = q_current[idx] + delta_wrapped

    return q_fixed


def _orientation_error_vector(target_vec: np.ndarray, current_vec: np.ndarray) -> np.ndarray:
    """Return the minimal rotation vector (axis * angle) to align current_vec with target_vec."""
    v0 = current_vec / np.linalg.norm(current_vec)
    v1 = target_vec / np.linalg.norm(target_vec)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    angle = np.arccos(dot)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.cross(v0, v1)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v0[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, v0) * v0
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / axis_norm
    return axis * angle

class LevenbergMarquardtIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        arm_qpos_adrs: list[int],
        arm_qvel_adrs: list[int],
        arm_joint_limits: list[tuple[float, float]],
        continuous_indices: list[int] | None = None,
        max_iters: int = 50, # Cap at 50 to ensure real-time performance
        pos_tol: float = 0.001,
        ori_weight: float = IK_ORIENTATION_WEIGHT,
        initial_lambda: float = 0.1,
    ):
        self.model = model
        self.data = data
        self.ee_site_id = ee_site_id
        self.arm_qpos_adrs = arm_qpos_adrs
        self.arm_qvel_adrs = arm_qvel_adrs
        self.joint_limits = arm_joint_limits
        self.continuous_indices = continuous_indices or []
        self.n_arm = len(arm_qpos_adrs)

        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.ori_weight = ori_weight
        self.initial_lambda = initial_lambda

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.eye = np.eye(self.n_arm)

        self.stall_threshold = 1e-6
        self.stall_patience = 5
        self.last_iterations = 0

    def solve(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None = None,
        q_seed: np.ndarray | None = None,
        active_dof: int | None = None,
    ) -> np.ndarray:
        if active_dof is None:
            active_dof = self.n_arm
        target_pos = np.asarray(target_pos, dtype=float)
        if target_vec is not None:
            target_vec = np.asarray(target_vec, dtype=float)
            target_vec = target_vec / np.linalg.norm(target_vec)

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        q_current = np.array([self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float)
        seeds = self._build_seeds(q_seed)

        if active_dof < self.n_arm:
            for s in seeds:
                s[active_dof:] = q_current[active_dof:]

        results = []
        for seed in seeds:
            q, iters, perr, oerr = self._solve_single(target_pos, target_vec, seed, active_dof)
            
            q = shortest_joint_configuration(q_current, q, self.continuous_indices)
            
            results.append((q.copy(), iters, perr, oerr))

        valid_results = [r for r in results if r[2] < 0.01]
        if valid_results:
            valid_results.sort(key=lambda r: r[3])
            best_ori_err = valid_results[0][3]
            top_results = [r for r in valid_results if r[3] <= best_ori_err + 0.0174] # 1 degree
            
            # Heavier weight on joint-space distance to prevent configuration flips
            j_weights = np.array([5.0, 2.0, 2.0, 1.0])
            top_results.sort(
                key=lambda r: joint_distance(r[0] * j_weights, q_current * j_weights, self.continuous_indices)
            )
            best_q, best_iters, best_pos_err, best_ori_err = top_results[0]
        else:
            results.sort(key=lambda r: r[2])
            best_q, best_iters, best_pos_err, best_ori_err = results[0]

        self.last_iterations = best_iters

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)

        return best_q

    def _solve_single(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None,
        q_start: np.ndarray,
        active_dof: int,
    ) -> tuple[np.ndarray, int, float, float]:
        q_arm = q_start.copy()
        for i_arm, adr in enumerate(self.arm_qpos_adrs):
            self.data.qpos[adr] = q_arm[i_arm]
        mujoco.mj_forward(self.model, self.data)

        lam = self.initial_lambda
        v = 2.0
        best_err = float('inf')
        stalls = 0

        for it in range(self.max_iters):
            ee_pos = np.array(self.data.site_xpos[self.ee_site_id], dtype=float)
            ee_xmat = np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(3, 3)
            current_vec = ee_xmat[:, 0]

            pos_err_vec = target_pos - ee_pos
            pos_err_norm = float(np.linalg.norm(pos_err_vec))

            if target_vec is not None:
                ori_err_vec = _orientation_error_vector(target_vec, current_vec)
                ori_err_norm = float(np.linalg.norm(ori_err_vec))
                residual = np.concatenate([pos_err_vec, self.ori_weight * ori_err_vec])
                cost = 0.5 * np.sum(residual**2)
            else:
                ori_err_norm = 0.0
                residual = pos_err_vec
                cost = 0.5 * np.sum(residual**2)

            if target_vec is None:
                if pos_err_norm < self.pos_tol:
                    return q_arm, it + 1, pos_err_norm, 0.0
            else:
                if pos_err_norm < self.pos_tol and ori_err_norm < 0.05:
                    return q_arm, it + 1, pos_err_norm, ori_err_norm

            if cost >= best_err:
                lam *= v
                v *= 2.0
            else:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * (best_err - cost) / cost)**3)
                v = 2.0
                best_err = cost

            if cost > 0 and abs(best_err - cost) < self.stall_threshold:
                stalls += 1
                if stalls >= self.stall_patience:
                    break
            else:
                stalls = 0

            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.ee_site_id)
            cols = self.arm_qvel_adrs[:active_dof]
            J = self.jacp[:, cols]
            if target_vec is not None:
                J = np.vstack([J, self.ori_weight * self.jacr[:, cols]])

            H = J.T @ J
            g = J.T @ residual
            H_lm = H + lam * np.eye(active_dof)

            import numpy.linalg as nla
            try:
                dq = nla.solve(H_lm, g)
            except nla.LinAlgError:
                dq = nla.lstsq(H_lm, g, rcond=None)[0]

            q_arm[:active_dof] = q_arm[:active_dof] + dq

            for j in range(self.n_arm):
                lo, hi = self.joint_limits[j]
                if np.isfinite(lo) or np.isfinite(hi):
                    q_arm[j] = float(np.clip(q_arm[j], lo, hi))

            for i_arm, adr in enumerate(self.arm_qpos_adrs):
                self.data.qpos[adr] = q_arm[i_arm]
            mujoco.mj_forward(self.model, self.data)

        ee_pos = np.array(self.data.site_xpos[self.ee_site_id], dtype=float)
        ee_xmat = np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(3, 3)
        pos_err_norm = float(np.linalg.norm(target_pos - ee_pos))
        ori_err_norm = 0.0
        if target_vec is not None:
            ori_err_vec = _orientation_error_vector(target_vec, ee_xmat[:, 0])
            ori_err_norm = float(np.linalg.norm(ori_err_vec))

        return q_arm, self.max_iters, pos_err_norm, ori_err_norm

    def _build_seeds(self, q_seed: np.ndarray | None) -> list[np.ndarray]:
        """Return a simplified set of seeds to favor the current configuration."""
        q_current = np.array([self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float)
        y0 = float(q_current[0])
        yaw_bases = [y0, y0 + 1.57, y0 - 1.57, y0 + 3.14]
        if q_seed is not None:
            y_seed = float(q_seed[0])
            yaw_bases = [y_seed, y_seed + 1.57, y_seed - 1.57, y_seed + 3.14]
        yaw_bases = [(y + 3.14159) % (2 * 3.14159) - 3.14159 for y in yaw_bases]

        pitch_vals = [1.5, -1.5, 3.14, -3.14]
        seeds = []
        for yaw in yaw_bases:
            for p1 in pitch_vals:
                for p2 in pitch_vals:
                    s = q_current.copy()
                    s[0], s[1], s[2] = yaw, p1, p2
                    # Note: We leave s[3] as it is (q_current[3]) to avoid unwanted jumps
                    seeds.append(s)

        import random as _rng
        for _ in range(50):
            s = q_current.copy()
            for j in range(self.n_arm):
                lo, hi = self.joint_limits[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    s[j] = _rng.uniform(lo, hi)
                else:
                    s[j] = _rng.uniform(-3.14, 3.14)
            seeds.append(s)

        if q_seed is not None:
            seeds.insert(0, np.array(q_seed[:self.n_arm], dtype=float))
            
        return seeds

class KinovaMuJoCoBackend(KinovaBackend):
    """MuJoCo backend with computed-torque control and jerk-limited trajectories."""

    # ── Nominal trajectory limits (per-joint: J1, J2, J3, J4) ────────
    _V_MAX_ARM = np.array([1.0, 1.0, 1.0, 1.0])          # rad/s
    _A_MAX_ARM = np.array([2.0, 2.0, 2.0, 2.0])        # rad/s²
    _J_MAX_ARM = np.array([40.0, 40.0, 40.0, 40.0])    # rad/s³

    # Trajectory shaping bandwidth (per-joint).  Independent of limits so
    # governor scaling v_max/a_max cannot accidentally increase omega.
    # Rule of thumb: omega ≈ a_max / v_max for a well-tuned baseline.
    _OMEGA_ARM = np.array([1.5, 2.0, 2.0, 2.0])           # rad/s

    # ── Computed-torque gains (acceleration space) ───────────────────
    _KP_ARM = np.array([500.0, 300.0, 300.0, 300.0])         # rad/s²/rad
    _KD_ARM = np.array([10.0,  30.0, 30.0,  30.0])         # rad/s²/(rad/s)

    # ── Finger PD gains (torque space) ─────────────────────────────
    _KP_FINGER = 5.0
    _KD_FINGER = 0.005
    # "none"=no bias, "gravity_only"=gravity only (hard to isolate in MuJoCo),
    # "full_bias"=full qfrc_bias.  "none" is safest for small finger joints:
    # full_bias can introduce chatter from Coriolis coupling with arm motion.
    FINGER_BIAS_MODE = "none"

    # ── Control / physics rate ───────────────────────────────────────
    _CONTROL_DT = 0.001  # 1 kHz target control rate

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
        self._ct_controller: ComputedTorqueController | None = None
        self._finger_controller: PDController | None = None
        self._traj_gen: TrajectoryGenerator | None = None
        self._governor: ReferenceGovernor | None = None
        self._n_substeps: int = 2
        self._joint_ids: list[int] = []
        self._qpos_adr: list[int] = []
        self._qvel_adr: list[int] = []
        self._actuator_ids: list[int] = []
        self._joint_limits: list[tuple[float, float]] = []
        self._finger_indices: list[int] = []
        self._arm_indices: list[int] = []
        self._arm_dof_adrs: list[int] = []
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
        self._continuous_indices = []
        for idx, jid in enumerate(self._joint_ids):
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
            else:
                low, high = -np.inf, np.inf
                self._continuous_indices.append(idx)
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

        # ── Substep calculation (ceil ensures control_dt >= target) ────
        physics_dt = float(model.opt.timestep)
        self._n_substeps = max(1, math.ceil(self._CONTROL_DT / physics_dt))
        control_dt = self._n_substeps * physics_dt  # effective control dt

        # ── Initial joint state ──────────────────────────────────────
        q_init = np.array([float(data.qpos[adr]) for adr in self._qpos_adr], dtype=float)
        self._q_target_desired = q_init.copy()

        # ── Trajectory generator (arm joints only) ───────────────────
        n_arm = len(self._arm_indices)
        q_arm_init = np.array([q_init[i] for i in self._arm_indices], dtype=float)
        self._traj_gen = TrajectoryGenerator(
            n_joints=n_arm,
            v_max=self._V_MAX_ARM,
            a_max=self._A_MAX_ARM,
            j_max=self._J_MAX_ARM,
            omega=self._OMEGA_ARM,
            dt=control_dt,
        )
        self._traj_gen.reset(q_arm_init)

        # ── Computed torque controller (arm) ─────────────────────────
        self._ct_controller = ComputedTorqueController(
            Kp=self._KP_ARM,
            Kd=self._KD_ARM,
            arm_dof_adrs=self._arm_dof_adrs,
            nv=model.nv,
        )

        # ── Finger PD controller ─────────────────────────────────────
        n_fingers = len(self._finger_indices)
        self._finger_controller = PDController(
            Kp=np.full(n_fingers, self._KP_FINGER),
            Kd=np.full(n_fingers, self._KD_FINGER),
        )

        # ── Reference governor (J1) ──────────────────────────────────
        j1_limit = 25.0 # Enforce strict 25.0 Nm limit
        self._governor = ReferenceGovernor(torque_limit=j1_limit)
        self._j1_torque_limit = j1_limit

    def close(self) -> None:
        if self._env is None:
            return
        self._env.close()
        self._env = None
        self._ct_controller = None
        self._finger_controller = None
        self._traj_gen = None
        self._governor = None

    def reset_scene(self) -> None:
        """Reset the simulation scene and controller state."""
        env = self._require_env()
        env.reset()
        
        if self._initial_keyframe:
            env.set_model_keyframe(self._initial_keyframe)
            
        # Re-read initial state from data after reset
        q_init = np.array([float(env.data.qpos[adr]) for adr in self._qpos_adr], dtype=float)
        self._q_target_desired = q_init.copy()
        
        # Reset trajectory generator so the arm doesn't snap to the previous target
        q_arm_init = np.array([q_init[i] for i in self._arm_indices], dtype=float)
        self._traj_gen.reset(q_arm_init)

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
        q_curr = self.get_joint_angles_rad()
        
        
        
        # Decide if we got arm-only or full joint list
        n_arm = len(self._arm_indices)
        is_arm_only = len(q_des) == n_arm
        
        q_full = self._require_q_target_desired().copy()
        q_des_wrapped = list(q_des)

        # Wrap continuous joints for shortest path relative to current state
        for idx in self._continuous_indices:
            # Only wrap if the joint is actually in the provided q_des
            # (Continuous indices are relative to the full joint list)
            if idx < len(q_des_wrapped):
                diff = q_des_wrapped[idx] - q_curr[idx]
                q_des_wrapped[idx] -= 2 * np.pi * np.round(diff / (2 * np.pi))


        q_curr = np.array(self.get_joint_angles_rad(), dtype=float)

        if is_arm_only:
            j1_curr = q_curr[self._arm_indices[0]]
            j1_cmd = q_des_wrapped[0]
        else:
            j1_curr = q_curr[0]
            j1_cmd = q_des_wrapped[0]

        j1_raw_delta = j1_cmd - j1_curr
        j1_wrapped_delta = wrap_to_pi(j1_raw_delta)

        if is_arm_only:
            # Update only arm joints in the full target array
            for i_local, idx_in_full in enumerate(self._arm_indices):
                q_full[idx_in_full] = q_des_wrapped[i_local]
            target_arm = np.array(q_des_wrapped, dtype=float)
        else:
            # Use the full provided list (wrapped)
            q_full = np.array(q_des_wrapped, dtype=float)
            target_arm = np.array([q_full[i] for i in self._arm_indices], dtype=float)
        
        self._q_target_desired = self._clip_q(q_full)
        self._traj_gen.set_goal(target_arm)

    def step(
        self,
        pos_tol_rad: float = math.radians(0.8),
        vel_tol_rad_s: float = math.radians(2.0),
    ) -> bool:
        env = self._require_env()
        model = env.model
        data = env.data
        gov = self._governor
        traj = self._traj_gen

        # ══ a) Read sensors (once per control tick) ══════════════════
        q = np.array([data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([data.qvel[adr] for adr in self._qvel_adr], dtype=float)

        q_arm = np.array([q[i] for i in self._arm_indices], dtype=float)
        qd_arm = np.array([qd[i] for i in self._arm_indices], dtype=float)
        q_fingers = np.array([q[i] for i in self._finger_indices], dtype=float)
        qd_fingers = np.array([qd[i] for i in self._finger_indices], dtype=float)

        # ══ b) Apply governor scale from PREVIOUS tick to trajectory ══
        # scale_next was computed last tick; now apply it so the trajectory
        # uses the adjusted limits for its update.
        s = gov.scale_current  # committed last tick
        traj.set_limits(
            0,
            v_max=self._V_MAX_ARM[0] * s,
            a_max=self._A_MAX_ARM[0] * s,
        )

        # ══ c) Advance trajectory generator ═══════════════════════════
        q_des, qd_des, qdd_des = traj.update()
        
        # ── SANITY CHECKS ──
        if not (np.all(np.isfinite(q_des)) and np.all(np.isfinite(qd_des)) and np.all(np.isfinite(qdd_des))):
            raise ValueError(f"Trajectory generator emitted non-finite values: q_des={q_des}, qd_des={qd_des}, qdd_des={qdd_des}")
        
        if np.max(np.abs(qdd_des)) > 100.0:
            raise ValueError(f"Trajectory generator emitted absurd acceleration limits (max 100.0): qdd_des={qdd_des}")


        # ══ d) Computed torque controller (arm) — produces RAW torque ═
        tau_arm_raw, qdd_pd, qdd_cmd = self._ct_controller.compute(
            model, data,
            q_des, qd_des, qdd_des,
            q_arm, qd_arm,
        )
        tau_raw_j1 = float(tau_arm_raw[0])

        # ══ e) Update governor using RAW (pre-saturation) J1 torque ══
        # Returns (scale_current, scale_next).  scale_next will be
        # applied on the NEXT tick.
        gov_cur, gov_next = gov.update(tau_raw_j1)

        # ══ f) Finger PD ────────────────────────────────────────
        q_fingers_des = np.array(
            [self._q_target_desired[i] for i in self._finger_indices], dtype=float
        )
        tau_fingers_pd, _, _ = self._finger_controller.compute(
            q_fingers, qd_fingers, q_fingers_des
        )
        tau_fingers = np.array(tau_fingers_pd, dtype=float)
        # FINGER_BIAS_MODE: controls whether bias forces are added.
        # "none":         safest default — fingers are small and don't need
        #                 gravity comp; adding full bias risks Coriolis chatter.
        # "gravity_only": ideal but hard to isolate in MuJoCo (qfrc_bias
        #                 includes Coriolis).  Not implemented here.
        # "full_bias":    adds full qfrc_bias including Coriolis coupling
        #                 with arm motion — can cause chatter.
        if self.FINGER_BIAS_MODE == "full_bias":
            for fi, idx in enumerate(self._finger_indices):
                tau_fingers[fi] += float(data.qfrc_bias[self._qvel_adr[idx]])

        # ══ g) Assemble ctrl + saturation ─────────────────────────
        ctrl = np.zeros(model.nu, dtype=float)
        tau_sat_j1 = 0.0

        for ai, arm_idx in enumerate(self._arm_indices):
            actuator_id = self._actuator_ids[arm_idx]
            tau_val = float(tau_arm_raw[ai])
            if ai == 0:
                # J1: smooth tanh saturation to avoid hard-clip discontinuity
                limit = self._j1_torque_limit
                tau_sat = float(limit * np.tanh(tau_val / limit)) if limit > 0 else tau_val
                tau_sat_j1 = tau_sat
            else:
                tau_sat = self._clip_actuator_force(model, actuator_id, tau_val)
            ctrl[actuator_id] = tau_sat

        for fi, finger_idx in enumerate(self._finger_indices):
            actuator_id = self._actuator_ids[finger_idx]
            ctrl[actuator_id] = self._clip_actuator_force(
                model, actuator_id, float(tau_fingers[fi])
            )

        # ══ h) Substeps (N × mj_step + 1 viewer sync) ──────────────
        data.ctrl[:] = ctrl
        env.step_n(self._n_substeps)

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
        env = self._require_env()
        q_target_desired = self._require_q_target_desired()
        q = np.array([env.data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([env.data.qvel[adr] for adr in self._qvel_adr], dtype=float)
        # Check ARM joints only (Fix 3a)
        arm_pos_err = max(abs(q_target_desired[i] - q[i]) for i in self._arm_indices)
        arm_vel = max(abs(qd[i]) for i in self._arm_indices)
        return bool(arm_pos_err <= float(pos_tol_rad) and arm_vel <= float(vel_tol_rad_s))

    def get_joint_angles_rad(self) -> list[float]:
        self._require_env()
        return [float(self._env.data.qpos[adr]) for adr in self._qpos_adr]

    def get_target_joint_angles_rad(self) -> list[float]:
        """Read current target joint angles (arm only) in radians."""
        q_target = self._require_q_target_desired()
        # Return only the arm joint components as a list
        return [float(q_target[i]) for i in self._arm_indices]

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

    def get_finger_forces(self) -> dict:
        """Read current actuator forces for finger joints.

        Returns:
            dict with 'forces' (list[float]), 'max_abs_force' (float),
            and 'contact_detected' (bool) based on force threshold.
        """
        env = self._require_env()
        F_MAX = 10.0  # must match MJCF forcerange
        CONTACT_THRESHOLD = 0.8 * F_MAX

        forces = []
        for idx in self._finger_indices:
            act_id = self._actuator_ids[idx]
            forces.append(float(env.data.actuator_force[act_id]))

        max_abs = max(abs(f) for f in forces) if forces else 0.0

        # Contact detected when force is high AND finger velocity is near zero
        finger_vels = []
        for idx in self._finger_indices:
            vel_adr = self._qvel_adr[idx]
            finger_vels.append(abs(float(env.data.qvel[vel_adr])))
        max_vel = max(finger_vels) if finger_vels else 0.0

        contact_detected = max_abs >= CONTACT_THRESHOLD and max_vel < 0.1

        return {
            "forces": forces,
            "max_abs_force": round(max_abs, 4),
            "contact_detected": contact_detected,
        }



    def _get_ik_solver(self):
        """
        Lazily construct the IK solver with correct arm-local continuous joints.
        """

        if not hasattr(self, "_ik_solver") or self._ik_solver is None:

            # Convert continuous indices from full joint indexing → arm indexing
            arm_continuous_indices = [
                i_arm
                for i_arm, idx_full in enumerate(self._arm_indices)
                if idx_full in self._continuous_indices
            ]
            print("full continuous:", self._continuous_indices)
            print("arm continuous:", arm_continuous_indices)

            self._ik_solver = LevenbergMarquardtIK(
                model=self._require_env().model,
                data=self._require_env().data,
                ee_site_id=self._ee_site_id,
                arm_qpos_adrs=[self._qpos_adr[i] for i in self._arm_indices],
                arm_qvel_adrs=[self._qvel_adr[i] for i in self._arm_indices],
                arm_joint_limits=[self._joint_limits[i] for i in self._arm_indices],
                continuous_indices=arm_continuous_indices,
            )

        return self._ik_solver

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
        solver = self._get_ik_solver()
        best_q = solver.solve(np.array(target_pos), target_x, q_seed, active_dof=active_dof)
        self._last_ik_iterations = solver.last_iterations
        return [float(x) for x in best_q]

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for position only (no orientation constraint)."""
        active_dof = self.arm_dof if move_wrist else self.arm_dof - 1
        solver = self._get_ik_solver()
        best_q = solver.solve(np.array(target_pos), None, q_seed, active_dof=active_dof)
        self._last_ik_iterations = solver.last_iterations
        return [float(x) for x in best_q]

    def solve_ik_z_down(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        """Solve IK for position accuracy while pointing straight down."""
        active_dof = self.arm_dof if move_wrist else self.arm_dof - 1
        solver = self._get_ik_solver()
        # Downward vector is -Z in world coords
        best_q = solver.solve(np.array(target_pos), np.array([0.0, 0.0, -1.0]), q_seed, active_dof=active_dof)
        self._last_ik_iterations = solver.last_iterations
        return [float(x) for x in best_q]


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

    def _require_traj_gen(self) -> TrajectoryGenerator:
        if self._traj_gen is None:
            raise RuntimeError("KinovaMuJoCoBackend.init() must be called before use.")
        return self._traj_gen

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
