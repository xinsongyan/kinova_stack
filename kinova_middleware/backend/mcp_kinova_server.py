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
    mjpython kinova_middleware/backend/mcp_kinova_server.py

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
import numpy as np

if __package__ in (None, ""):
    # Preserve direct-script execution from the repo root while using package imports.
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from kinova_middleware.backend.controller import KinovaController
from kinova_middleware.backend.factory import build_backend
from kinova_middleware.backend.mcp_server.tool_registry import setup_tools
from kinova_middleware.backend.mcp_server.toolsets.task_prompts import setup_prompts
from kinova_middleware.scenes.scene_selector import resolve_scene_number, select_scene

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
KINOVA_MODE: str = os.getenv("KINOVA_MODE", "sim").strip().lower()
TARGET_SPEED: float = float(os.getenv("KINOVA_TARGET_SPEED_RAD_S", "2.0"))
CONTROL_HZ: float = float(os.getenv("CONTROL_HZ", "500"))
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
_current_scene_path: str | None = None
_mcp_configured = False

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


def _build_controller_for_scene(scene_path: str) -> KinovaController:
    """Create and initialise a controller for the requested scene."""
    log.info(
        "Initialising Kinova controller  mode=%s  speed=%.2f rad/s",
        KINOVA_MODE,
        TARGET_SPEED,
    )

    if KINOVA_MODE == "sim":
        backend = build_backend(
            "sim",
            scene_path=scene_path,
            viewer=True,
            target_speed_rad_s=TARGET_SPEED,
            ee_site="ee_marker",
        )
    elif KINOVA_MODE == "real":
        backend = build_backend("real")
    else:
        raise ValueError(f"KINOVA_MODE must be 'sim' or 'real', got '{KINOVA_MODE}'")

    controller = KinovaController(backend)
    controller.init()
    log.info("Controller ready.  DOF=%d  arm_dof=%d", controller.dof, controller.arm_dof)
    return controller


def _move_home_after_init() -> None:
    """Move the freshly initialised controller to its home configuration."""
    ctrl = _get_controller()
    log.info("Moving home …")
    ctrl.move_home()
    _run_until_reached(timeout_s=10.0, hold_seconds=HOLD_SECONDS)
    log.info("Home reached. Simulation frozen.")


def _reset_or_reload_scene(scene_number: int | None = None) -> dict[str, Any]:
    """Reset the current scene or hot-swap to another scene in sim mode."""
    global _controller, _current_scene_path

    if scene_number is None:
        ctrl = _get_controller()
        with _physics_lock:
            ctrl.reset_scene()
        scene_path = _current_scene_path
        return {
            "status": "ok",
            "message": "Scene reset successfully.",
            "scene_changed": False,
            "scene_name": os.path.basename(scene_path) if scene_path else None,
            "scene_path": scene_path,
        }

    resolved_scene_path = resolve_scene_number(scene_number)
    scene_name = os.path.basename(resolved_scene_path)

    if KINOVA_MODE != "sim":
        raise RuntimeError("Scene switching by number is only supported in sim mode.")

    if _current_scene_path == resolved_scene_path and _controller is not None:
        with _physics_lock:
            _controller.reset_scene()
        return {
            "status": "ok",
            "message": f"Scene reset successfully: {scene_name}.",
            "scene_changed": False,
            "scene_number": int(scene_number),
            "scene_name": scene_name,
            "scene_path": resolved_scene_path,
        }

    log.info("Switching scene to #%d: %s", int(scene_number), scene_name)
    with _physics_lock:
        if _controller is not None:
            _controller.close()
            _controller = None

        _controller = _build_controller_for_scene(resolved_scene_path)
        _current_scene_path = resolved_scene_path

    _move_home_after_init()
    return {
        "status": "ok",
        "message": f"Scene switched and reset successfully: {scene_name}.",
        "scene_changed": True,
        "scene_number": int(scene_number),
        "scene_name": scene_name,
        "scene_path": resolved_scene_path,
    }


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
    
    # ── Trajectory Tracking ─────────────────────────────────────────
    with _physics_lock:
        q_start = np.array(ctrl.get_joint_angles_rad()[:ctrl.arm_dof])
        # Capture starting Cartesian position (position only)
        pos_tuple, _ = ctrl.get_end_effector_pose()
        p_start = np.array(pos_tuple)
        q_target = np.array(ctrl.get_target_joint_angles_rad())
    
    dist_direct = float(np.linalg.norm(q_target - q_start))
    accum_dist = 0.0
    q_prev = q_start.copy()
    
    # Trackers for 1/4, 2/4, 3/4 progress
    milestones = [0.25, 0.50, 0.75]
    milestone_states = {} # milestone -> (q_state, p_state)
    
    # Default to looser tolerance if not specified (5 deg pos, 10 deg/s vel)
    # Default to looser tolerance if not specified
    if "pos_tol_rad" not in kwargs:
        kwargs["pos_tol_rad"] = 0.1  # ~5.7 deg
    if "vel_tol_rad_s" not in kwargs:
        kwargs["vel_tol_rad_s"] = 0.5  # ~28 deg/s

    while time.monotonic() < deadline:
        with _physics_lock:
            # Step the simulation and check if reached
            reached = ctrl.step(**kwargs)
            q_curr = np.array(ctrl.get_joint_angles_rad()[:ctrl.arm_dof])
            # Capture current Cartesian position
            pos_tuple, _ = ctrl.get_end_effector_pose()
            p_curr = np.array(pos_tuple)

        # Track accumulated travel
        step_dist = float(np.linalg.norm(q_curr - q_prev))
        accum_dist += step_dist
        q_prev = q_curr.copy()
        
        # Check milestones based on straight-line distance progress
        if dist_direct > 0.01:
            for m in milestones:
                if m not in milestone_states:
                    prog = float(np.linalg.norm(q_curr - q_start))
                    if prog >= m * dist_direct:
                        milestone_states[m] = (q_curr.copy(), p_curr.copy())

        if reached:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= hold_seconds:
                # Log final report
                _log_trajectory_report(q_start, p_start, q_target, q_curr, p_curr, milestone_states, dist_direct, accum_dist)
                return True
        else:
            settled_since = None

        time.sleep(dt)

    # Log timeout report
    with _physics_lock:
        q_final = np.array(ctrl.get_joint_angles_rad()[:ctrl.arm_dof])
        pos_tuple, _ = ctrl.get_end_effector_pose()
        p_final = np.array(pos_tuple)
    _log_trajectory_report(q_start, p_start, q_target, q_final, p_final, milestone_states, dist_direct, accum_dist, timed_out=True)
    return False


def _log_trajectory_report(q_start, p_start, q_target, q_end, p_end, milestones, dist_direct, dist_accum, timed_out=False):
    """Print a detailed joint-space and Cartesian trajectory report to the log."""
    def fmt_q(q):
        return "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"
    
    def fmt_p(p):
        return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"
    
    label = "TIMEOUT" if timed_out else "REACHED"
    log.info("── Trajectory Report [%s] ──────────────────", label)
    log.info("  Start:  %s  %s", fmt_q(q_start), fmt_p(p_start))
    
    if 0.25 in milestones: 
        q, p = milestones[0.25]
        log.info("  1/4:    %s  %s", fmt_q(q), fmt_p(p))
    if 0.50 in milestones: 
        q, p = milestones[0.50]
        log.info("  1/2:    %s  %s", fmt_q(q), fmt_p(p))
    if 0.75 in milestones: 
        q, p = milestones[0.75]
        log.info("  3/4:    %s  %s", fmt_q(q), fmt_p(p))
        
    log.info("  End:    %s  %s", fmt_q(q_end), fmt_p(p_end))
    log.info("  Target: %s", fmt_q(q_target))
    log.info("  Distances: Direct=%.4f, Traveled=%.4f (ratio=%.2f)", 
             dist_direct, dist_accum, dist_accum/dist_direct if dist_direct > 1e-6 else 1.0)
    log.info("──────────────────────────────────────────────────")


def _quat_rotation_error(
    q1: tuple[float, ...], q2: tuple[float, ...]
) -> float:
    """Rotation error magnitude (rad) between two unit quaternions (xyzw)."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


# Stepper-thread removed: simulation now only advances during tool calls.


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _configure_mcp() -> None:
    global _mcp_configured
    if _mcp_configured:
        return

    capabilities = _get_controller().supported_capabilities()
    tool_names = setup_tools(
        mcp,
        {
            "get_controller": _get_controller,
            "motion_lock": _motion_lock,
            "physics_lock": _physics_lock,
            "run_until_reached": _run_until_reached,
            "reset_or_reload_scene": _reset_or_reload_scene,
            "capabilities": capabilities,
        },
    )
    prompt_names = setup_prompts(mcp, capabilities=capabilities)
    _mcp_configured = True
    log.info("Registered MCP tools: %s", ", ".join(tool_names) if tool_names else "(none)")
    log.info("Registered MCP prompts: %s", ", ".join(prompt_names) if prompt_names else "(none)")


def _startup() -> None:
    global _controller, _current_scene_path

    # ── Scene selection (must happen before backend creation) ─────────
    scene_path = select_scene()
    _current_scene_path = scene_path
    log.info("Selected scene: %s", os.path.basename(scene_path))
    _controller = _build_controller_for_scene(scene_path)
    _configure_mcp()
    _move_home_after_init()


def _shutdown() -> None:
    global _controller, _current_scene_path
    log.info("Shutting down …")
    _stepper_running.clear()
    if _controller is not None:
        _controller.close()
        _controller = None
    _current_scene_path = None
    log.info("Controller closed.")


atexit.register(_shutdown)


mcp = FastMCP(
    "Kinova Arm Controller",
    instructions=(
        "MCP server for controlling a Kinova Gen3 Lite arm. "
        "All motion tools block until the target is reached."
    ),
)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    print("[mcp_kinova_server] Starting …", file=sys.stderr, flush=True)
    _startup()
    print("[mcp_kinova_server] Controller ready – launching MCP server.", file=sys.stderr, flush=True)
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
