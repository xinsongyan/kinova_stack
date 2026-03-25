#!/usr/bin/env python3
"""
Test: Pick up a cube and hold it for 60 seconds (headless).

Directly uses the KinovaMuJoCoBackend (no MCP server) to:
  1. Load single_cube.xml scene (headless)
  2. Pick up the cube using the same sequence as demo scripts
  3. Hold for 60 s of sim-time, logging diagnostics every 5 s
  4. Report PASS/FAIL

Diagnostics logged: cube Z, finger forces, arm joint errors.
Goal: diagnose why blocks slip out of the gripper.

Usage:
    cd kinova_stack
    .venv/bin/mjpython kinova_middleware/demo_scripts/test_hold_cube_60s.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import mujoco
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "backend"))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))

for p in (_BACKEND_DIR, _ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from kinova_mujoco_backend import KinovaMuJoCoBackend  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
SCENE_PATH = os.path.join(_THIS_DIR, "..", "scenes", "single_cube.xml")
CUBE_BODY_NAME = "cube_1"

# Grasp parameters (from grab_shapes.py / demo_reach_cube.py)
GRIP_OPEN = 1.0       # fully open
GRIP_CLOSE = 0.58     # box grip (from grab_shapes.py)
GRASP_CLEARANCE = 0.009  # extra Z offset above cube top (from grab_shapes.py)

# Hold duration
HOLD_SIM_SECONDS = 60.0
LOG_INTERVAL_S = 5.0

# Thresholds
CUBE_Z_DROP_THRESHOLD = 0.05  # cube must stay above initial_z + this

# ── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str, indent: int = 0):
    print(f"{'  ' * indent}{msg}", flush=True)


def get_cube_pos(backend: KinovaMuJoCoBackend) -> np.ndarray:
    """Read cube position from MuJoCo data."""
    env = backend._env
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
    return env.data.xpos[bid].copy()


def run_until_reached(backend: KinovaMuJoCoBackend, timeout_s: float = 15.0,
                       hold_s: float = 0.3, hz: float = 500.0) -> bool:
    """Step the backend until target is reached and held stable, or timeout."""
    dt = 1.0 / hz
    deadline = time.monotonic() + timeout_s
    settled_since = None
    kwargs = {"pos_tol_rad": 0.1, "vel_tol_rad_s": 0.5}

    while time.monotonic() < deadline:
        reached = backend.step(**kwargs)
        if reached:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= hold_s:
                return True
        else:
            settled_since = None
        time.sleep(dt)
    return False


def step_sim_seconds(backend: KinovaMuJoCoBackend, seconds: float, hz: float = 500.0):
    """Step the simulation for a given number of wall-clock seconds."""
    dt = 1.0 / hz
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        backend.step()
        time.sleep(dt)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  CUBE GRIP HOLD TEST (60 seconds)")
    log("=" * 60)

    # ── 1. Init backend (headless) ──────────────────────────────────────
    log("\n[1] Initialising MuJoCo backend (headless) …")
    backend = KinovaMuJoCoBackend(
        model_path=os.path.abspath(SCENE_PATH),
        viewer=False,
        ee_site="ee_marker",
    )
    backend.init()
    env = backend._env
    model = env.model
    data = env.data
    log(f"  Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

    # ── 2. Home ─────────────────────────────────────────────────────────
    log("\n[2] Moving home …")
    backend.move_home()
    run_until_reached(backend, timeout_s=10.0)
    log("  Home reached.")

    # ── 3. Open gripper ─────────────────────────────────────────────────
    log("\n[3] Opening gripper …")
    backend.set_gripper_percent(GRIP_OPEN)
    step_sim_seconds(backend, 1.0)
    log("  Gripper open.")

    # ── 4. Get cube pose ────────────────────────────────────────────────
    cube_pos = get_cube_pos(backend)
    cx, cy, cz = cube_pos
    log(f"\n[4] Cube at ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    # Compute grasp height (box: half-height is 0.03 in single_cube.xml)
    cube_half_z = 0.03  # from XML: size="0.03 0.03 0.03"
    top_z = cz + cube_half_z
    approach_z = top_z + 0.15
    grasp_z = top_z + GRASP_CLEARANCE

    log(f"  top_z={top_z:.3f}, approach_z={approach_z:.3f}, grasp_z={grasp_z:.3f}")

    # ── 5. Face the cube (rotate J1 toward cube) ────────────────────────
    log("\n[5] Facing cube (J1 rotation) …")
    theta_world = math.atan2(cy, cx)
    j1_target = -theta_world
    j1_target = math.atan2(math.sin(j1_target), math.cos(j1_target))

    q_arm = backend.get_joint_angles_rad()[:4]
    q_arm[0] = j1_target
    backend.send_joint_position_rad(q_arm)
    run_until_reached(backend, timeout_s=10.0)
    log(f"  J1 set to {math.degrees(j1_target):.1f}°")

    # ── 6. Approach (position-only IK) ──────────────────────────────────
    approach_pos = [cx, cy, approach_z]
    log(f"\n[6] Approaching {approach_pos} …")
    q_approach = backend.solve_ik_position_only(approach_pos)
    backend.send_joint_position_rad(q_approach)
    reached = run_until_reached(backend, timeout_s=15.0)
    ee_pos, _ = backend.get_end_effector_pose()
    pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, approach_pos)))
    log(f"  Approach {'OK' if reached else 'TIMEOUT'}  pos_err={pos_err:.4f}m")

    # ── 7. Descend to grasp height ──────────────────────────────────────
    grasp_pos = [cx, cy, grasp_z]
    log(f"\n[7] Descending to {grasp_pos} …")
    current_q = backend.get_joint_angles_rad()
    q_descend = backend.solve_ik_position_only(grasp_pos, q_seed=current_q)
    backend.send_joint_position_rad(q_descend)
    reached = run_until_reached(backend, timeout_s=15.0)
    ee_pos, _ = backend.get_end_effector_pose()
    pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, grasp_pos)))
    log(f"  Descend {'OK' if reached else 'TIMEOUT'}  pos_err={pos_err:.4f}m")

    # ── 8. Grasp ────────────────────────────────────────────────────────
    log(f"\n[8] Grasping (grip={GRIP_CLOSE*100:.0f}%) …")
    backend.set_gripper_percent(GRIP_CLOSE)
    step_sim_seconds(backend, 1.5)  # let fingers settle

    finger_info = backend.get_finger_forces()
    log(f"  Finger forces: max={finger_info['max_abs_force']:.2f}N  contact={finger_info['contact_detected']}")

    # ── 9. Lift ─────────────────────────────────────────────────────────
    lift_z = approach_z  # lift back to approach height
    lift_pos = [cx, cy, lift_z]
    log(f"\n[9] Lifting to {lift_pos} …")
    current_q = backend.get_joint_angles_rad()
    q_lift = backend.solve_ik_position_only(lift_pos, q_seed=current_q)
    backend.send_joint_position_rad(q_lift)
    reached = run_until_reached(backend, timeout_s=15.0)

    # Check if cube was actually lifted
    cube_after_lift = get_cube_pos(backend)
    log(f"  Lift {'OK' if reached else 'TIMEOUT'}  cube_z: {cz:.3f} → {cube_after_lift[2]:.3f}")

    if cube_after_lift[2] < cz + CUBE_Z_DROP_THRESHOLD:
        log(f"\n  ✗ FAIL — Cube was NOT lifted (Δz={cube_after_lift[2]-cz:.3f}m)")
        log("  Aborting hold test.")
        backend.close()
        sys.exit(1)

    log(f"  ✓ Cube lifted successfully (Δz={cube_after_lift[2]-cz:.3f}m)")

    # ── 10. HOLD for 60 seconds ─────────────────────────────────────────
    log(f"\n[10] Holding cube for {HOLD_SIM_SECONDS:.0f}s …")
    log(f"     {'time':>6s}  {'cube_z':>8s}  {'Δz':>8s}  {'f_max':>6s}  {'contact':>7s}  {'arm_err':>8s}")
    log(f"     {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*8}")

    hold_start_z = cube_after_lift[2]
    min_z = hold_start_z
    hold_passed = True
    n_logs = int(HOLD_SIM_SECONDS / LOG_INTERVAL_S)

    for i in range(n_logs):
        # Step for LOG_INTERVAL_S seconds of wall-clock time
        step_sim_seconds(backend, LOG_INTERVAL_S)

        # Read diagnostics
        cube_pos_now = get_cube_pos(backend)
        cube_z_now = cube_pos_now[2]
        delta_z = cube_z_now - hold_start_z
        min_z = min(min_z, cube_z_now)

        finger_info = backend.get_finger_forces()
        f_max = finger_info["max_abs_force"]
        contact = finger_info["contact_detected"]

        # Arm joint tracking error
        q_current = np.array(backend.get_joint_angles_rad()[:4])
        q_target = np.array(backend.get_target_joint_angles_rad())
        arm_err = float(np.max(np.abs(q_target - q_current)))

        elapsed = (i + 1) * LOG_INTERVAL_S
        status = "✓" if cube_z_now > cz + CUBE_Z_DROP_THRESHOLD else "✗"
        log(f"  {status} {elapsed:5.0f}s  {cube_z_now:8.4f}  {delta_z:+8.4f}  {f_max:6.2f}  {str(contact):>7s}  {arm_err:8.4f}")

        if cube_z_now < cz + CUBE_Z_DROP_THRESHOLD:
            log(f"\n  ✗ CUBE DROPPED at t={elapsed:.0f}s!")
            hold_passed = False
            break

    # ── 11. Final report ────────────────────────────────────────────────
    log("\n" + "=" * 60)
    if hold_passed:
        log(f"  ✓ PASS — Cube held for {HOLD_SIM_SECONDS:.0f}s")
        log(f"    min cube Z = {min_z:.4f}m  (threshold = {cz + CUBE_Z_DROP_THRESHOLD:.4f}m)")
    else:
        log(f"  ✗ FAIL — Cube dropped before {HOLD_SIM_SECONDS:.0f}s")
        log(f"    min cube Z = {min_z:.4f}m  (threshold = {cz + CUBE_Z_DROP_THRESHOLD:.4f}m)")
    log("=" * 60)

    # ── Cleanup ─────────────────────────────────────────────────────────
    backend.set_gripper_percent(GRIP_OPEN)
    step_sim_seconds(backend, 1.0)
    backend.close()

    sys.exit(0 if hold_passed else 1)


if __name__ == "__main__":
    main()
