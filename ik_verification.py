#!/usr/bin/env python3
"""IK Verification Benchmark for Kinova MICO (m1n4s300).

Implements Levenberg–Marquardt inverse kinematics with MuJoCo Jacobians,
then moves the arm to each target site using the kinova_sim controllers.
Measures position error, orientation error, and downward-alignment.

Algorithm:
    dq = (JᵀJ + λI)⁻¹ Jᵀ e

    where J is the combined [jacp ; w·jacr] Jacobian,
    e is [e_pos ; w·e_rot]  (position + weighted orientation error),
    and λ adapts per step:  λ↑ on error increase, λ↓ on decrease.

Run:  mjpython ik_verification.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Imports from kinova_sim
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "kinova_sim")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from sim_env import SimEnv                      # noqa: E402
from controller import ComputedTorqueController # noqa: E402
from trajectory import TrajectoryGenerator      # noqa: E402
from governor import ReferenceGovernor          # noqa: E402


# ===========================================================================
# Constants
# ===========================================================================
SCENE_PATH = os.path.join(_HERE, "kinova_middleware", "scenes", "ik_verification.xml")

ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4")
FINGER_JOINTS = (
    "joint_finger_1", "joint_finger_tip_1",
    "joint_finger_2", "joint_finger_tip_2",
    "joint_finger_3", "joint_finger_tip_3",
)
EE_SITE_NAME = "ee_marker"

# Desired EE orientation: gripper pointing straight down.
# The ee_marker site **x-axis** (red) points OUT of the hand.
# To point it down (world -Z), we apply a +90° rotation around Y:
# w = cos(45°), y = sin(45°)
DOWN_QUAT = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # wxyz

# Motion controller params
V_MAX = np.array([1.0, 1.0, 1.0, 1.0])
A_MAX = np.array([2.0, 2.0, 2.0, 2.0])
J_MAX = np.array([20.0, 150.0, 150.0, 150.0])
KP = np.array([500.0, 800.0, 800.0, 400.0])
KD = np.array([10.0, 100.0, 100.0, 50.0])
CONTROL_DT = 0.001
J1_TORQUE_LIMIT = 25.0
MOTION_TIMEOUT_S = 10.0
SETTLE_HOLD_S = 0.5

# Thresholds
POS_ERROR_THRESHOLD = 0.01       # 10 mm
ORI_ERROR_THRESHOLD_DEG = 5.0    # 5 degrees
DOWNWARD_DOT_THRESHOLD = 0.1     # Allow up to +0.2 (slightly pointing up) based on user feedback

CSV_OUTPUT = os.path.join(_HERE, "ik_validation_results.csv")


# ===========================================================================
# Data classes
# ===========================================================================
@dataclass
class TargetResult:
    name: str = ""
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reached_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos_error: float = 0.0
    ori_error_deg: float = 0.0
    downward_dot: float = 0.0
    ik_iterations: int = 0
    success: bool = False
    failure_reason: str = ""


# ===========================================================================
# Quaternion helpers  (all wxyz convention)
# ===========================================================================
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion (w,x,y,z)."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product  a ⊗ b  (wxyz)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (wxyz)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def orientation_error_vector(v_current: np.ndarray, v_target: np.ndarray) -> np.ndarray:
    """Compute shortest-path rotation vector to align v_current with v_target.

    Both vectors must be unit vectors.
    Returns an axis-angle vector (3,) in world frame representing the rotation.
    This provides 2 orientation constraints, leaving rotation around v_current unconstrained.
    """
    cos_theta = np.clip(np.dot(v_current, v_target), -1.0, 1.0)
    
    # Singularity at 180 degrees
    if cos_theta < -0.999:
        # Pick arbitrary orthogonal axis
        if abs(v_current[0]) < 0.9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, v_current) * v_current
        axis = axis / np.linalg.norm(axis)
        return axis * math.pi
        
    axis = np.cross(v_current, v_target)
    axis_len = np.linalg.norm(axis)
    
    if axis_len < 1e-8:
        return np.zeros(3)
        
    axis = axis / axis_len
    angle = math.acos(cos_theta)
    return axis * angle


# ===========================================================================
# Levenberg–Marquardt IK Solver
# ===========================================================================
class LevenbergMarquardtIK:
    """Damped least-squares IK with adaptive Levenberg–Marquardt damping.

    Supports position-only or position+orientation solving.

    Update rule per iteration:
        dq = (JᵀJ + λI)⁻¹ Jᵀ e

    Adaptive λ:
        - If error decreased: λ *= λ_down   (get closer to Gauss-Newton)
        - If error increased: λ *= λ_up     (fall back to gradient descent)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        arm_joint_ids: list[int],
        arm_qpos_adrs: list[int],
        arm_dof_adrs: list[int],
        joint_limits: list[tuple[float, float]],
        *,
        step_size: float = 1.0,
        tol_pos: float = 1e-3,
        tol_ori: float = 0.05,     # ~3 degrees
        lambda_init: float = 1.0,
        lambda_up: float = 10.0,
        lambda_down: float = 0.1,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e6,
        max_iters: int = 200,
        ori_weight: float = 1.0,
        stall_threshold: float = 1e-8,
        stall_patience: int = 10,
    ) -> None:
        self.model = model
        self.data = data
        self.ee_site_id = ee_site_id
        self.arm_joint_ids = arm_joint_ids
        self.arm_qpos_adrs = arm_qpos_adrs
        self.arm_dof_adrs = arm_dof_adrs
        self.joint_limits = joint_limits
        self.n_arm = len(arm_joint_ids)

        # Tuning
        self.step_size = step_size
        self.tol_pos = tol_pos
        self.tol_ori = tol_ori
        self.lambda_init = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_iters = max_iters
        self.ori_weight = ori_weight
        self.stall_threshold = stall_threshold
        self.stall_patience = stall_patience

        # Output
        self.last_iterations = 0

    # ----- public interface -----
    def solve(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None = None,
        q_seed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve IK. Returns arm joint angles.

        Args:
            target_pos: desired EE position  (3,)
            target_vec: desired EE downward vector (3,) or None for pos-only
            q_seed: optional initial joint seed  (n_arm,)
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if target_vec is not None:
            target_vec = np.asarray(target_vec, dtype=float)
            target_vec = target_vec / np.linalg.norm(target_vec)

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        q_current = np.array([self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float)

        seeds = self._build_seeds(q_seed)

        # Evaluate all seeds
        results: list[tuple[np.ndarray, int, float, float]] = []
        for seed in seeds:
            q, iters, perr, oerr = self._solve_single(target_pos, target_vec, seed)
            results.append((q.copy(), iters, perr, oerr))

        # Select best by position first, then orientation, then joint-space proximity
        # 1. Filter seeds that achieved the position tolerance (or close to it)
        valid_results = [r for r in results if r[2] < POS_ERROR_THRESHOLD]
        if valid_results:
            # Among geometrically valid positions, find the best orientation
            valid_results.sort(key=lambda r: r[3])
            best_ori_err = valid_results[0][3]
            
            # Keep all solutions that are within 1.0 degree of the absolute best orientation
            top_results = [r for r in valid_results if r[3] <= best_ori_err + math.radians(1.0)]
            
            # Tie-break by joint-space distance from CURRENT joint state (heavily penalize base joint rotation)
            j_weights = np.array([2.0, 1.0, 1.0, 0.5])  # Joint 1 is heavily weighted
            top_results.sort(key=lambda r: np.linalg.norm((r[0] - q_current) * j_weights))
            
            best_q, best_iters, best_pos_err, best_ori_err = top_results[0]
        else:
            # If none reached position, just pick the one with lowest position error
            results.sort(key=lambda r: r[2])
            best_q, best_iters, best_pos_err, best_ori_err = results[0]

        self.last_iterations = best_iters

        # Restore simulation
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)

        return best_q

    # ----- LM core -----
    def _solve_single(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None,
        q_start: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float]:
        """Run LM from a single seed. Returns (q_arm, iters, pos_err, ori_err)."""
        model, data = self.model, self.data
        q_arm = q_start.copy()
        qpos_save = data.qpos.copy()

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)

        use_ori = target_vec is not None and self.ori_weight > 0
        n_rows = 6 if use_ori else 3

        lam = self.lambda_init
        prev_err = float("inf")
        stall_count = 0
        iters = 0

        for it in range(self.max_iters):
            iters = it + 1

            # Forward kinematics
            data.qpos[:] = qpos_save
            for i, adr in enumerate(self.arm_qpos_adrs):
                data.qpos[adr] = q_arm[i]
            mujoco.mj_forward(model, data)

            # --- Compute error ---
            pos = data.site_xpos[self.ee_site_id].copy()
            e_pos = target_pos - pos

            if use_ori:
                R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)
                v_x = R_ee[:, 0]
                e_ori = orientation_error_vector(v_x, target_vec)
                e = np.concatenate([e_pos, self.ori_weight * e_ori])
            else:
                e = e_pos

            err_norm = float(np.linalg.norm(e))

            # Check convergence
            pos_err = float(np.linalg.norm(e_pos))
            if use_ori:
                ori_err = float(np.linalg.norm(e_ori))
                converged = pos_err <= self.tol_pos and ori_err <= self.tol_ori
            else:
                converged = pos_err <= self.tol_pos

            if converged:
                break

            # Stall detection
            improvement = prev_err - err_norm
            if abs(improvement) < self.stall_threshold:
                stall_count += 1
                if stall_count >= self.stall_patience:
                    break
            else:
                stall_count = 0

            # Adaptive λ
            if err_norm < prev_err:
                lam = max(lam * self.lambda_down, self.lambda_min)
            else:
                lam = min(lam * self.lambda_up, self.lambda_max)

            prev_err = err_norm

            # --- Jacobian ---
            mujoco.mj_jacSite(model, data, jacp, jacr, self.ee_site_id)
            Jp = jacp[:, self.arm_dof_adrs].copy()

            if use_ori:
                Jr = jacr[:, self.arm_dof_adrs].copy()
                J = np.vstack([Jp, self.ori_weight * Jr])
            else:
                J = Jp

            # --- LM step:  dq = (JᵀJ + λI)⁻¹ Jᵀ e ---
            JtJ = J.T @ J
            Jte = J.T @ e
            A = JtJ + lam * np.eye(self.n_arm)

            try:
                dq = np.linalg.solve(A, Jte)
            except np.linalg.LinAlgError:
                # Fallback to least-squares
                dq, _, _, _ = np.linalg.lstsq(A, Jte, rcond=None)

            q_arm = q_arm + self.step_size * dq

            # Clamp to joint limits
            for j in range(self.n_arm):
                lo, hi = self.joint_limits[j]
                if np.isfinite(lo) or np.isfinite(hi):
                    q_arm[j] = float(np.clip(q_arm[j], lo, hi))

        # --- Final evaluation ---
        data.qpos[:] = qpos_save
        for i, adr in enumerate(self.arm_qpos_adrs):
            data.qpos[adr] = q_arm[i]
        mujoco.mj_forward(model, data)

        pos = data.site_xpos[self.ee_site_id].copy()
        final_pos_err = float(np.linalg.norm(pos - target_pos))

        final_ori_err = 0.0
        if use_ori:
            R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)
            v_x = R_ee[:, 0]
            e_ori = orientation_error_vector(v_x, target_vec)
            final_ori_err = float(np.linalg.norm(e_ori))

        # Reject floor-piercing solutions
        for gi in range(1, model.ngeom):
            if data.geom_xpos[gi][2] < 0.001:
                final_pos_err = 999.0
                break

        return q_arm, iters, final_pos_err, final_ori_err

    # ----- seed generation -----
    def _build_seeds(self, q_seed: np.ndarray | None) -> list[np.ndarray]:
        """Generate diverse seed configurations."""
        seeds: list[np.ndarray] = []
        q_current = np.array(
            [self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float
        )

        yaw_bases = [0.0, math.pi/2, -math.pi/2, math.pi]
        if q_seed is not None:
            y0 = float(q_seed[0])
            yaw_bases = [y0, y0+math.pi/2, y0-math.pi/2, y0+math.pi]
        yaw_bases = [(y+math.pi) % (2*math.pi) - math.pi for y in yaw_bases]

        # pitch values spanning the joint ranges
        pitch_vals = [1.5, -1.5, 3.14, -3.14, 3.23, 4.90, 2.5, 5.5]

        # Systematic grid: 4 yaw × 8 p1 × 8 p2 = 256 seeds
        for yaw in yaw_bases:
            for p1 in pitch_vals:
                for p2 in pitch_vals:
                    s = q_current.copy()
                    s[0] = yaw
                    s[1] = p1
                    s[2] = p2
                    s[3] = 0.5
                    seeds.append(s)

        # Random seeds
        for _ in range(50):
            s = q_current.copy()
            for j in range(self.n_arm):
                lo, hi = self.joint_limits[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    s[j] = random.uniform(lo, hi)
                else:
                    s[j] = random.uniform(-math.pi, math.pi)
            seeds.append(s)

        # Explicit seed first
        if q_seed is not None:
            seeds.insert(0, np.array(q_seed[:self.n_arm], dtype=float))

        return seeds


# ===========================================================================
# Evaluation helpers
# ===========================================================================
def compute_orientation_error_deg(ee_z: np.ndarray, desired: np.ndarray) -> float:
    ee_z = ee_z / (np.linalg.norm(ee_z) + 1e-12)
    d = desired / (np.linalg.norm(desired) + 1e-12)
    cos_a = float(np.clip(np.dot(ee_z, d), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def get_ee_state(model, data, site_id):
    """Return (pos, rot_matrix, z_axis) for the EE site."""
    pos = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, R, R[:, 2].copy()


# ===========================================================================
# Motion Controller
# ===========================================================================
class MotionController:
    """Wraps kinova_sim controllers for joint-space motion."""

    def __init__(self, model, data, viewer, arm_qpos_adrs, arm_dof_adrs,
                 arm_actuator_ids, finger_actuator_ids):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.arm_qpos_adrs = arm_qpos_adrs
        self.arm_dof_adrs = arm_dof_adrs
        self.arm_actuator_ids = arm_actuator_ids
        self.finger_actuator_ids = finger_actuator_ids
        self.n_arm = len(arm_qpos_adrs)

        physics_dt = float(model.opt.timestep)
        self.n_substeps = max(1, math.ceil(CONTROL_DT / physics_dt))

        q0 = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)
        control_dt = self.n_substeps * physics_dt

        self.traj = TrajectoryGenerator(
            n_joints=self.n_arm, v_max=V_MAX, a_max=A_MAX,
            j_max=J_MAX, dt=control_dt,
        )
        self.traj.reset(q0)
        self.ct = ComputedTorqueController(
            Kp=KP, Kd=KD, arm_dof_adrs=arm_dof_adrs, nv=model.nv,
        )
        self.governor = ReferenceGovernor(torque_limit=J1_TORQUE_LIMIT)

    def move_to(self, q_target, timeout_s=MOTION_TIMEOUT_S):
        self.traj.set_goal(q_target)
        start = self.data.time
        settled_since = None

        while (self.data.time - start) < timeout_s:
            reached = self._step_once()
            if reached:
                if settled_since is None:
                    settled_since = self.data.time
                elif (self.data.time - settled_since) >= SETTLE_HOLD_S:
                    return True
            else:
                settled_since = None
        return False

    def _step_once(self):
        model, data = self.model, self.data
        q_arm = np.array([data.qpos[a] for a in self.arm_qpos_adrs])
        qd_arm = np.array([data.qvel[a] for a in self.arm_dof_adrs])

        s = self.governor.scale_current
        self.traj.set_limits(0, v_max=V_MAX[0]*s, a_max=A_MAX[0]*s)

        q_des, qd_des, qdd_des = self.traj.update()
        tau_arm, _, _ = self.ct.compute(model, data, q_des, qd_des, qdd_des, q_arm, qd_arm)
        self.governor.update(float(tau_arm[0]))

        tau_arm[0] = J1_TORQUE_LIMIT * np.tanh(tau_arm[0] / J1_TORQUE_LIMIT)
        for i in range(1, self.n_arm):
            aid = self.arm_actuator_ids[i]
            if model.actuator_forcelimited[aid]:
                lo = float(model.actuator_forcerange[aid, 0])
                hi = float(model.actuator_forcerange[aid, 1])
                tau_arm[i] = float(np.clip(tau_arm[i], lo, hi))

        ctrl = np.zeros(model.nu, dtype=float)
        for i in range(self.n_arm):
            ctrl[self.arm_actuator_ids[i]] = tau_arm[i]
        data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(model, data)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                sys.exit("Viewer closed.")

        q_now = np.array([data.qpos[a] for a in self.arm_qpos_adrs])
        qd_now = np.array([data.qvel[a] for a in self.arm_dof_adrs])
        q_goal = self.traj.q_goal
        pos_err = max(abs(q_goal[i] - q_now[i]) for i in range(self.n_arm))
        vel = max(abs(qd_now[i]) for i in range(self.n_arm))
        return bool(pos_err <= math.radians(0.8) and vel <= math.radians(2.0))


# ===========================================================================
# Site discovery
# ===========================================================================
def discover_target_sites(model):
    targets = []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and name.startswith("target_"):
            targets.append((name, sid))
    random.shuffle(targets)
    return targets


# ===========================================================================
# Reporting
# ===========================================================================
def print_results_table(results):
    header = (
        f"{'Target':<12} {'Target Pos':>24} {'Reached Pos':>24} "
        f"{'Pos Err':>8} {'Ori Err':>8} {'Down Dot':>9} "
        f"{'IK Iter':>7} {'Status':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        tp = f"({r.target_pos[0]:+.3f}, {r.target_pos[1]:+.3f}, {r.target_pos[2]:+.3f})"
        rp = f"({r.reached_pos[0]:+.3f}, {r.reached_pos[1]:+.3f}, {r.reached_pos[2]:+.3f})"
        st = "PASS" if r.success else "FAIL"
        print(
            f"{r.name:<12} {tp:>24} {rp:>24} "
            f"{r.pos_error:>8.4f} {r.ori_error_deg:>7.2f}° {r.downward_dot:>+9.4f} "
            f"{r.ik_iterations:>7d} {st:>8}"
        )
        if not r.success:
            print(f"{'':>12} ↳ {r.failure_reason}")
    print(sep)


def print_summary(results):
    n = len(results)
    ok = sum(1 for r in results if r.success)
    pos_errs = [r.pos_error for r in results]
    ori_errs = [r.ori_error_deg for r in results]
    failed = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print("  IK VERIFICATION — FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Targets tested:         {n}")
    print(f"  Successful IK solves:   {ok} / {n}")
    print(f"  Average position error: {np.mean(pos_errs):.4f} m")
    print(f"  Maximum position error: {np.max(pos_errs):.4f} m")
    print(f"  Maximum orient. error:  {np.max(ori_errs):.2f}°")
    if failed:
        print(f"\n  FAILED TARGETS ({len(failed)}):")
        for r in failed:
            print(f"    - {r.name}: {r.failure_reason}")
    else:
        print("\n  ✓ All targets passed!")
    print(f"{'='*60}")


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "target_name",
            "target_x", "target_y", "target_z",
            "reached_x", "reached_y", "reached_z",
            "pos_error_m", "ori_error_deg", "downward_dot",
            "ik_iterations", "success", "failure_reason",
        ])
        for r in results:
            w.writerow([
                r.name,
                f"{r.target_pos[0]:.6f}", f"{r.target_pos[1]:.6f}", f"{r.target_pos[2]:.6f}",
                f"{r.reached_pos[0]:.6f}", f"{r.reached_pos[1]:.6f}", f"{r.reached_pos[2]:.6f}",
                f"{r.pos_error:.6f}", f"{r.ori_error_deg:.4f}", f"{r.downward_dot:.6f}",
                r.ik_iterations, r.success, r.failure_reason,
            ])
    print(f"\n  Results written to: {path}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    print("=" * 60)
    print("  IK VERIFICATION — Levenberg–Marquardt Solver")
    print("  Robot: Kinova MICO m1n4s300  (4 DOF)")
    print("  Scene: ik_verification.xml")
    print("=" * 60)

    if not os.path.isfile(SCENE_PATH):
        print(f"ERROR: Scene not found: {SCENE_PATH}")
        return 1

    print(f"\n  Loading scene: {SCENE_PATH}")
    env = SimEnv(SCENE_PATH, viewer=True)
    model, data = env.model, env.data
    env.set_model_keyframe("home")
    mujoco.mj_forward(model, data)

    # --- Resolve joint IDs ---
    arm_jnt_ids, arm_qpos_adrs, arm_dof_adrs = [], [], []
    arm_actuator_ids, arm_limits = [], []

    for jname in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"Joint '{jname}' not found"
        arm_jnt_ids.append(jid)
        arm_qpos_adrs.append(int(model.jnt_qposadr[jid]))
        arm_dof_adrs.append(int(model.jnt_dofadr[jid]))
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jname}")
        assert aid >= 0, f"Actuator 'motor_{jname}' not found"
        arm_actuator_ids.append(aid)
        if model.jnt_limited[jid]:
            arm_limits.append((float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])))
        else:
            arm_limits.append((-np.inf, np.inf))

    finger_actuator_ids = []
    for jname in FINGER_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jname}")
        if aid >= 0:
            finger_actuator_ids.append(aid)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    assert ee_site_id >= 0, f"EE site '{EE_SITE_NAME}' not found"

    print(f"  Arm joints: {ARM_JOINTS}")
    print(f"  EE site: {EE_SITE_NAME} (id={ee_site_id})")

    # --- Discover targets ---
    targets = discover_target_sites(model)
    assert targets, "No target_* sites found"
    print(f"  Discovered {len(targets)} target sites:")
    for name, sid in targets:
        p = data.site_xpos[sid]
        print(f"    {name}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    # --- Create LM IK solver ---
    ik_solver = LevenbergMarquardtIK(
        model=model, data=data,
        ee_site_id=ee_site_id,
        arm_joint_ids=arm_jnt_ids,
        arm_qpos_adrs=arm_qpos_adrs,
        arm_dof_adrs=arm_dof_adrs,
        joint_limits=arm_limits,
        step_size=1.0,
        tol_pos=5e-4,       # 0.5 mm
        tol_ori=0.05,        # ~3 degree
        lambda_init=1.0,
        max_iters=200,
        ori_weight=0.01,     # very low weight so position is strictly prioritized
    )

    # --- Create motion controller ---
    motion = MotionController(
        model, data, env.viewer,
        arm_qpos_adrs, arm_dof_adrs,
        arm_actuator_ids, finger_actuator_ids,
    )

    # --- Move home ---
    print("\n  Setting to HOME position...")
    q_home = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)
    # Set home instantly
    for i, adr in enumerate(arm_qpos_adrs):
        data.qpos[adr] = q_home[i]
    mujoco.mj_forward(model, data)
    print("  At HOME.\n")

    if env.viewer is not None:
        env.viewer.sync()

    # --- IK + motion loop ---
    results: list[TargetResult] = []
    prev_q: np.ndarray | None = None
    # ee_marker x-axis points OUT of the hand, so desired direction is DOWN
    desired_x_down = np.array([0.0, 0.0, -1.0])

    for idx, (tgt_name, tgt_sid) in enumerate(targets):
        result = TargetResult(name=tgt_name)

        mujoco.mj_forward(model, data)
        tgt_pos = data.site_xpos[tgt_sid].copy()
        result.target_pos = tgt_pos.copy()

        print(f"  [{idx+1}/{len(targets)}] {tgt_name}: "
              f"pos=({tgt_pos[0]:+.3f}, {tgt_pos[1]:+.3f}, {tgt_pos[2]:+.3f})")

        # Solve IK with position + downward orientation
        print("    Solving IK...", end=" ", flush=True)
        t0 = time.monotonic()
        q_ik = ik_solver.solve(tgt_pos, desired_x_down, q_seed=prev_q)
        dt_ik = time.monotonic() - t0
        result.ik_iterations = ik_solver.last_iterations
        print(f"done ({result.ik_iterations} iters, {dt_ik:.2f}s)")

        # Move arm via controller
        print("    Moving arm...", end=" ", flush=True)
        t0 = time.monotonic()
        reached = motion.move_to(q_ik)
        dt_move = time.monotonic() - t0
        if reached:
            print(f"done ({dt_move:.2f}s)")
        else:
            print(f"timed out ({dt_move:.2f}s)")

        prev_q = q_ik.copy()

        # Measure errors
        mujoco.mj_forward(model, data)
        ee_pos, ee_R, ee_z = get_ee_state(model, data, ee_site_id)
        ee_x = ee_R[:, 0]  # x-axis points out of gripper
        result.reached_pos = ee_pos.copy()
        result.pos_error = float(np.linalg.norm(tgt_pos - ee_pos))
        result.ori_error_deg = compute_orientation_error_deg(ee_x, desired_x_down)
        result.downward_dot = float(ee_x[2])  # negative = gripper down

        # Evaluate pass/fail
        fails = []
        if result.pos_error > POS_ERROR_THRESHOLD:
            fails.append(f"pos_err={result.pos_error:.4f}m > {POS_ERROR_THRESHOLD}m")
        if result.downward_dot > DOWNWARD_DOT_THRESHOLD:
            fails.append(f"down_dot={result.downward_dot:+.4f} > {DOWNWARD_DOT_THRESHOLD} (gripper points too high)")

        result.success = len(fails) == 0
        result.failure_reason = "; ".join(fails)

        tag = "✓ PASS" if result.success else "✗ FAIL"
        print(f"    {tag}  pos_err={result.pos_error:.4f}m  "
              f"ori_err={result.ori_error_deg:.1f}°  "
              f"down_dot={result.downward_dot:+.4f}")
        if not result.success:
            print(f"    ↳ {result.failure_reason}")

        results.append(result)
        prev_q = q_ik

    # --- Report ---
    print_results_table(results)
    print_summary(results)
    write_csv(results, CSV_OUTPUT)

    # Keep viewer open
    print("\n  Close the MuJoCo viewer to exit.")
    if env.viewer is not None:
        while env.viewer.is_running():
            mujoco.mj_step(model, data)
            env.viewer.sync()
            time.sleep(0.01)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
