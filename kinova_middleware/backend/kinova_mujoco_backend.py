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
IK_ORIENTATION_WEIGHT = 0.3


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

        # ── CSV logging (at control rate) ────────────────────────────
        self._j1_log_file = open("j1_tracking_log.csv", "w")
        self._j1_log_file.write(
            "time,"
            "q_des_j1,q_j1,err_j1,"
            "qd_des_j1,qd_j1,"
            "qdd_des_j1,qdd_cmd_j1,"
            "tau_raw_j1,tau_sat_j1,"
            "util_raw_j1,util_ema_j1,"
            "gov_scale_cur,gov_scale_next,"
            "braking_j1\n"
        )

    def close(self) -> None:
        if hasattr(self, "_j1_log_file") and self._j1_log_file:
            self._j1_log_file.close()
            self._j1_log_file = None
        if self._env is None:
            return
        self._env.close()
        self._env = None
        self._ct_controller = None
        self._finger_controller = None
        self._traj_gen = None
        self._governor = None

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
        # Update trajectory generator goal
        if self._traj_gen is not None:
            q_arm_goal = np.array([self._q_target_desired[i] for i in self._arm_indices])
            self._traj_gen.set_goal(q_arm_goal)

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

        # ══ i) Log at control rate ─────────────────────────────────
        if hasattr(self, "_j1_log_file") and self._j1_log_file:
            self._j1_log_file.write(
                f"{data.time:.6f},"
                f"{q_des[0]:.6f},{q_arm[0]:.6f},{q_des[0]-q_arm[0]:.6f},"
                f"{qd_des[0]:.6f},{qd_arm[0]:.6f},"
                f"{qdd_des[0]:.6f},{qdd_cmd[0]:.6f},"
                f"{tau_raw_j1:.6f},{tau_sat_j1:.6f},"
                f"{gov.util_raw:.4f},{gov.util_ema:.4f},"
                f"{gov_cur:.4f},{gov_next:.4f},"
                f"{int(traj.braking[0])}\n"
            )

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
