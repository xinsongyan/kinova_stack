#!/usr/bin/env python3
from __future__ import annotations
# --- Fast-start: disable beartype's slow AST-transformation import hooks ----
# The dependency chain  fastmcp → docket → key_value → beartype.claw  installs
# AST-rewriting import hooks that add 90+ seconds to first startup on Py 3.11.
# key_value checks this env var and uses O0 (no-op) strategy when it is set.
import os as _os
_os.environ["PY_KEY_VALUE_DISABLE_BEARTYPE"] = "true"
# ---------------------------------------------------------------------------
"""
FastMCP server for Kinova arm control.

Exposes synchronous, blocking MCP tools that drive the Kinova arm through the
KinovaController / SafetyWrapperBackend stack.  When the server starts it
initializes the controller (MuJoCo sim by default), opens the viewer, and
begins a continuous stepping loop.  Motion tool calls block until the target
is reached and held stable, or a timeout fires.

Launch (sim):
    mjpython kinova_middleware/mcp_kinova_server.py

Env vars:
    KINOVA_MODE             "sim" (default) | "real"
    KINOVA_TARGET_SPEED_RAD_S   smooth-target speed (default 1.5)
    CONTROL_HZ              stepping frequency     (default 200)
    HOLD_SECONDS            hold-stable duration   (default 0.4)
    COMMAND_TIMEOUT_S       motion timeout         (default 30.0)
"""

import atexit
import logging
import math
import os
import sys
import threading
import time
from typing import Any

from fastmcp import FastMCP

# ── project imports (on PYTHONPATH via the run script) ──────────────────────
from kinova_controller import KinovaController
from kinova_mujoco_backend import KinovaMuJoCoBackend
from kinova_backend import CartesianPose

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
KINOVA_MODE: str = os.getenv("KINOVA_MODE", "sim").strip().lower()
TARGET_SPEED: float = float(os.getenv("KINOVA_TARGET_SPEED_RAD_S", "1.5"))
CONTROL_HZ: float = float(os.getenv("CONTROL_HZ", "200"))
HOLD_SECONDS: float = float(os.getenv("HOLD_SECONDS", "0.4"))
COMMAND_TIMEOUT_S: float = float(os.getenv("COMMAND_TIMEOUT_S", "30.0"))

log = logging.getLogger("mcp_kinova")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
    force=True,
)

# ---------------------------------------------------------------------------
# Controller singleton  (initialised in _startup, used by every tool)
# ---------------------------------------------------------------------------
_controller: KinovaController | None = None

# Only one motion command may run at a time.
_motion_lock = threading.Lock()

# Protects *all* calls into the controller / MuJoCo physics.
# The stepper thread holds this briefly each tick; motion handlers hold it
# for the duration of their blocking loop so the stepper pauses.
_physics_lock = threading.Lock()

# Stepper-thread control
_stepper_running = threading.Event()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_controller() -> KinovaController:
    """Return the initialised controller or raise."""
    if _controller is None:
        raise RuntimeError("Controller not initialised – server not started?")
    return _controller


def _run_until_reached(
    timeout_s: float = COMMAND_TIMEOUT_S,
    hold_seconds: float = HOLD_SECONDS,
    hz: float = CONTROL_HZ,
    **kwargs: Any,
) -> bool:
    """Step the controller until target is reached and held, or timeout.

    Must be called while the caller already holds ``_motion_lock``.
    Acquires ``_physics_lock`` per-tick so the stepper thread pauses.

    Returns True if reached, False if timed out.
    """
    ctrl = _get_controller()
    dt = 1.0 / hz
    deadline = time.monotonic() + timeout_s
    settled_since: float | None = None
    
    # Default to looser tolerance if not specified (5 deg pos, 10 deg/s vel)
    if "pos_tol_rad" not in kwargs:
        kwargs["pos_tol_rad"] = math.radians(5.0)
    if "vel_tol_rad_s" not in kwargs:
        kwargs["vel_tol_rad_s"] = math.radians(10.0)

    while time.monotonic() < deadline:
        with _physics_lock:
            reached = ctrl.step(**kwargs)

        if reached:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= hold_seconds:
                return True
        else:
            settled_since = None

        time.sleep(dt)

    return False  # timeout


def _quat_rotation_error(
    q1: tuple[float, ...], q2: tuple[float, ...]
) -> float:
    """Rotation error magnitude (rad) between two unit quaternions (xyzw)."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


# ---------------------------------------------------------------------------
# Background stepper thread
# ---------------------------------------------------------------------------

def _stepper_loop() -> None:
    """Run controller.step() at CONTROL_HZ while no motion command owns the lock."""
    dt = 1.0 / CONTROL_HZ
    while _stepper_running.is_set():
        acquired = _physics_lock.acquire(timeout=dt)
        if acquired:
            try:
                if _controller is not None:
                    _controller.step()
            finally:
                _physics_lock.release()
        time.sleep(dt)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _startup() -> None:
    global _controller

    log.info("Initialising Kinova controller  mode=%s  speed=%.2f rad/s", KINOVA_MODE, TARGET_SPEED)

    if KINOVA_MODE == "sim":
        backend = KinovaMuJoCoBackend(viewer=True, target_speed_rad_s=TARGET_SPEED, ee_site="ee_marker")
    elif KINOVA_MODE == "real":
        from kinova_sdk_backend import KinovaSDKBackend
        backend = KinovaSDKBackend()
    else:
        raise ValueError(f"KINOVA_MODE must be 'sim' or 'real', got '{KINOVA_MODE}'")

    _controller = KinovaController(backend)
    _controller.init()
    log.info("Controller ready.  DOF=%d  arm_dof=%d", _controller.dof, _controller.arm_dof)

    # Move home and wait for it
    log.info("Moving home …")
    _controller.move_home()
    _run_until_reached(timeout_s=10.0, hold_seconds=HOLD_SECONDS)
    log.info("Home reached.")

    # Start background stepper
    _stepper_running.set()
    t = threading.Thread(target=_stepper_loop, daemon=True, name="kinova-stepper")
    t.start()
    log.info("Stepper thread started at %.0f Hz", CONTROL_HZ)


def _shutdown() -> None:
    global _controller
    log.info("Shutting down …")
    _stepper_running.clear()
    time.sleep(0.05)  # let stepper finish current tick
    if _controller is not None:
        _controller.close()
        _controller = None
    log.info("Controller closed.")


atexit.register(_shutdown)


mcp = FastMCP(
    "Kinova Arm Controller",
    instructions=(
        "MCP server for controlling a Kinova Gen3 Lite arm. "
        "All motion tools block until the target is reached."
    ),
)

# ── Tool 1: move_home ──────────────────────────────────────────────────────

@mcp.tool()
def move_home() -> dict:
    """Command the arm to its home configuration and block until reached.

    Returns:
        status: "ok" or "timeout"
        message: human-readable result
    """
    log.info("Tool  move_home()")
    ctrl = _get_controller()
    with _motion_lock:
        ctrl.move_home()
        reached = _run_until_reached()
    status = "ok" if reached else "timeout"
    msg = "Home reached." if reached else "Timed out moving home."
    log.info("  → %s", msg)
    return {"status": status, "message": msg}


# ── Tool 2: get_end_effector_pose ──────────────────────────────────────────

@mcp.tool()
def get_end_effector_pose() -> dict:
    """Read the current end-effector Cartesian pose (non-blocking).

    Returns:
        position: {x, y, z}  (metres)
        quaternion: {qx, qy, qz, qw}
    """
    ctrl = _get_controller()
    with _physics_lock:
        pos, quat = ctrl.get_end_effector_pose()
    return {
        "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
        "quaternion": {"qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3]},
    }


# ── Tool 3: get_joint_state ───────────────────────────────────────────────

@mcp.tool()
def get_joint_state() -> dict:
    """Read current joint angles (rad) and velocities (rad/s) (non-blocking).

    Returns:
        q_rad: list of joint angles
        qd_rad_s: list of joint velocities
    """
    ctrl = _get_controller()
    with _physics_lock:
        q = ctrl.get_joint_angles_rad()
        qd = ctrl.get_joint_vel_rad()
    return {"q_rad": q, "qd_rad_s": qd}


# ── Tool 4: set_gripper ──────────────────────────────────────────────────

@mcp.tool()
def set_gripper(percent: float) -> dict:
    """Set gripper opening. 1.0 = fully open, 0.0 = fully closed.

    Blocks briefly to let the gripper settle.

    Args:
        percent: opening ratio in [0.0, 1.0]

    Returns:
        status: "ok"
        message: description
    """
    log.info("Tool  set_gripper(%.2f)", percent)
    ctrl = _get_controller()
    p = max(0.0, min(1.0, float(percent)))
    with _motion_lock:
        ctrl.set_gripper_percent(p)
        _run_until_reached(timeout_s=5.0, hold_seconds=0.3)
    msg = f"Gripper set to {p*100:.0f}%."
    log.info("  → %s", msg)
    return {"status": "ok", "message": msg}


# ── Tool 5: move_joints ──────────────────────────────────────────────────

@mcp.tool()
def move_joints(
    q: list[float],
    units: str = "deg",
) -> dict:
    """Move the arm joints to target angles and block until reached.

    Accepts arm joints only (length must equal arm_dof).

    Args:
        q: target joint values (arm joints only)
        units: "deg" or "rad" (default "deg")

    Returns:
        status: "ok" | "timeout"
        reached: bool
        final_q_rad: final joint angles in radians
    """
    ctrl = _get_controller()
    n = ctrl.arm_dof
    log.info("Tool  move_joints(%s, units=%s)", q, units)

    if len(q) != n:
        return {
            "status": "error",
            "message": f"Expected {n} arm joints, got {len(q)}.",
        }

    if units.lower().startswith("d"):
        q_rad = [math.radians(float(v)) for v in q]
    else:
        q_rad = [float(v) for v in q]

    with _motion_lock:
        ctrl.send_joint_position_rad(q_rad)
        reached = _run_until_reached()

    with _physics_lock:
        final_q = ctrl.get_joint_angles_rad()

    status = "ok" if reached else "timeout"
    log.info("  → %s  reached=%s", status, reached)
    return {"status": status, "reached": reached, "final_q_rad": final_q}


# ── Tool 6: move_pose ────────────────────────────────────────────────────

@mcp.tool()
def move_pose(
    target_pos: list[float],
    target_quat: list[float],
    seed_q_rad: list[float] | None = None,
    allow_orientation_fallback: bool = True,
) -> dict:
    """Move the end-effector to a Cartesian pose (IK → joint command → block).

    Args:
        target_pos: [x, y, z] in metres
        target_quat: [qx, qy, qz, qw] unit quaternion
        seed_q_rad: optional IK seed (arm joints)
        allow_orientation_fallback: if True and quaternion is invalid,
            fall back to position-only IK

    Returns:
        status: "ok" | "timeout" | "error"
        mode_used: "full_pose" | "position_only"
        q_target_rad: IK solution sent to arm
        final_pose: {position, quaternion}
        pos_err: Euclidean position error (m)
        rot_err: rotation error (rad), null if position-only
    """
    ctrl = _get_controller()
    log.info("Tool  move_pose(pos=%s, quat=%s)", target_pos, target_quat)

    if len(target_pos) != 3:
        return {"status": "error", "message": "target_pos must have 3 elements."}
    if len(target_quat) != 4:
        return {"status": "error", "message": "target_quat must have 4 elements."}

    # Validate quaternion
    quat_norm = math.sqrt(sum(v * v for v in target_quat))
    mode_used = "full_pose"

    if quat_norm < 1e-6:
        if allow_orientation_fallback:
            log.warning("  Quaternion near-zero → position-only fallback")
            # Use current orientation as a neutral quaternion for IK
            with _physics_lock:
                _, cur_quat = ctrl.get_end_effector_pose()
            target_quat = list(cur_quat)
            mode_used = "position_only"
        else:
            return {"status": "error", "message": "Invalid quaternion (near-zero norm)."}
    else:
        # Normalise
        target_quat = [v / quat_norm for v in target_quat]

    # Solve IK
    try:
        with _physics_lock:
            if mode_used == "position_only":
                q_target = ctrl.solve_ik_position_only(target_pos, seed_q_rad)
            else:
                q_target = ctrl.solve_ik(target_pos, target_quat, seed_q_rad)
    except ValueError as exc:
        log.error("  IK / safety error: %s", exc)
        return {"status": "error", "message": f"ik_failed: {exc}"}

    # Execute motion
    with _motion_lock:
        ctrl.send_joint_position_rad(q_target)
        reached = _run_until_reached()

    # Read final pose
    with _physics_lock:
        final_pos, final_quat = ctrl.get_end_effector_pose()

    # Compute errors
    pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(final_pos, target_pos)))
    rot_err: float | None = None
    if mode_used == "full_pose":
        rot_err = _quat_rotation_error(final_quat, tuple(target_quat))

    status = "ok" if reached else "timeout"
    log.info("  → %s  pos_err=%.4f m  rot_err=%s rad", status, pos_err, f"{rot_err:.4f}" if rot_err is not None else "n/a")
    return {
        "status": status,
        "mode_used": mode_used,
        "q_target_rad": q_target,
        "final_pose": {"position": list(final_pos), "quaternion": list(final_quat)},
        "pos_err": pos_err,
        "rot_err": rot_err,
    }


# ── Tool 7: rotate_wrist ──────────────────────────────────────────────────

@mcp.tool()
def rotate_wrist(angle_deg: float) -> dict:
    """Rotate the wrist (last arm joint) by a relative angle.

    Args:
        angle_deg: rotation angle in degrees (positive or negative)

    Returns:
        status: "ok" | "error"
        message: description of result
    """
    ctrl = _get_controller()
    log.info("Tool  rotate_wrist(%.1f deg)", angle_deg)
    
    with _motion_lock:
        try:
            ctrl.rotate_wrist(angle_deg)
            # The backend's rotate_wrist calls send_joint_position_rad, but we need to wait for it.
            # However, KinovaBackend.rotate_wrist just sends the command. It doesn't block.
            # So we should call _run_until_reached here.
            reached = _run_until_reached()
            status = "ok" if reached else "timeout"
        except Exception as e:
            log.error("rotate_wrist failed: %s", e)
            return {"status": "error", "message": str(e)}

    return {"status": status, "message": f"Rotate wrist by {angle_deg} deg finished. Reached={reached}"}


# ── Tool 8: get_object_pose ──────────────────────────────────────────────

@mcp.tool()
def get_object_pose(body_name: str) -> dict:
    """Read the Cartesian pose of any named body in the MuJoCo scene.

    Useful for locating objects like the cube, table, or any other body.

    Args:
        body_name: name of the body in the MJCF model (e.g. "cube")

    Returns:
        body_name: the queried body name
        position: {x, y, z}  (metres)
        quaternion: {qx, qy, qz, qw}
    """
    ctrl = _get_controller()
    log.info("Tool  get_object_pose(%s)", body_name)

    # Access the internal MuJoCo model/data through the backend
    backend = getattr(ctrl, "_backend", None)
    inner = getattr(backend, "_inner", backend)  # handle SafetyWrapper
    env = getattr(inner, "_env", None)
    if env is None:
        return {"status": "error", "message": "Cannot access MuJoCo environment."}

    import mujoco as _mj
    model = env.model
    data = env.data

    body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return {"status": "error", "message": f"Body '{body_name}' not found in model."}

    with _physics_lock:
        pos = data.xpos[body_id].copy()
        quat_wxyz = data.xquat[body_id].copy()

    # Convert MuJoCo wxyz → xyzw
    return {
        "body_name": body_name,
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "quaternion": {
            "qx": float(quat_wxyz[1]),
            "qy": float(quat_wxyz[2]),
            "qz": float(quat_wxyz[3]),
            "qw": float(quat_wxyz[0]),
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[mcp_kinova_server] Starting …", file=sys.stderr, flush=True)
    _startup()
    print("[mcp_kinova_server] Controller ready – launching MCP server.", file=sys.stderr, flush=True)
    mcp.run(transport="streamable-http")
