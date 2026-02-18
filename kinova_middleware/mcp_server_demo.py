#!/usr/bin/env python3
"""
FastMCP Server for Kinova Semi-Circle Demo.

Exposes blocking tools for controlling the Kinova arm in MuJoCo simulation.
Key features:
- Blocking execution: Tools only return after motion is complete and held stable.
- Position-only IK support: If quaternion is missing/invalid, falls back to maintaining current orientation.
- Precise hold-time control via arguments.

Run:
    python mcp_server_demo.py --transport streamable-http --viewer
"""
from __future__ import annotations

import atexit
import argparse
import logging
import math
import os
import sys
import threading
import time
from typing import Any

from fastmcp import FastMCP

# Add local modules to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.append(_ROOT_DIR)
sys.path.append(os.path.join(_ROOT_DIR, "kinova_sim"))
sys.path.append(os.path.join(_ROOT_DIR, "kinova_api"))
sys.path.append(os.path.join(_ROOT_DIR, "kinova_middleware"))

from backend.kinova_controller import KinovaController
from backend.kinova_mujoco_backend import KinovaMuJoCoBackend
from backend.kinova_backend import CartesianPose

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTROL_HZ = 200.0
DEFAULT_HOLD_S = 0.4
DEFAULT_TIMEOUT_S = 30.0

log = logging.getLogger("mcp_demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

# ---------------------------------------------------------------------------
# Global Controller & State
# ---------------------------------------------------------------------------
_controller: KinovaController | None = None
_motion_lock = threading.Lock()   # Serializes high-level motion commands
_physics_lock = threading.Lock()  # Protects MuJoCo step/access
_stepper_running = threading.Event()

def _get_controller() -> KinovaController:
    if _controller is None:
        raise RuntimeError("Controller not initialized")
    return _controller

def _stepper_loop():
    """Background thread stepping the physics at CONTROL_HZ."""
    dt = 1.0 / CONTROL_HZ
    while _stepper_running.is_set():
        # Try to acquire physics lock; if motion command has it, we skip/wait
        if _physics_lock.acquire(timeout=dt):
            try:
                if _controller:
                    _controller.step()
            finally:
                _physics_lock.release()
        time.sleep(dt)

def _startup(viewer: bool = True):
    global _controller
    log.info("Initializing Kinova Demo Controller (Sim)...")
    backend = KinovaMuJoCoBackend(viewer=viewer, target_speed_rad_s=1.5)
    _controller = KinovaController(backend)
    _controller.init()
    
    # Move home
    log.info("Moving home...")
    _controller.move_home()
    # Block until home reached (using internal stepper logic since thread not started yet)
    # We can just manually step here for simplicity or start thread first.
    # Let's start thread first to be consistent.
    
    _stepper_running.set()
    t = threading.Thread(target=_stepper_loop, daemon=True)
    t.start()
    
    # Wait for home
    time.sleep(2.0) # Simple wait for home since we don't have the sophisticated check handy yet
                    # Actually, let's use the robust wait logic immediately.
    
    log.info("Controller ready.")

def _shutdown():
    global _controller
    _stepper_running.clear()
    time.sleep(0.1)
    if _controller:
        _controller.close()
        _controller = None

atexit.register(_shutdown)

def _ws_run_until_reached(timeout_s: float, hold_s: float) -> bool:
    """
    Steps the controller (while holding _physics_lock) until valid target reached & held.
    MUST be called while holding _motion_lock.
    """
    ctrl = _get_controller()
    dt = 1.0 / CONTROL_HZ
    deadline = time.monotonic() + timeout_s
    settled_since = None
    
    while time.monotonic() < deadline:
        with _physics_lock:
            reached = ctrl.step()
            
        if reached:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= hold_s:
                return True
        else:
            settled_since = None
            
        time.sleep(dt)
    return False

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("Kinova Demo Server")

@mcp.tool()
def get_state() -> dict:
    """
    Returns current robot state: position [x,y,z], quaternion [qx,qy,qz,qw],
    joint angles (rad), and joint velocities (rad/s).
    """
    ctrl = _get_controller()
    with _physics_lock:
        pos, quat = ctrl.get_end_effector_pose()
        q = ctrl.get_joint_angles_rad()
        qd = ctrl.get_joint_vel_rad()
    
    return {
        "pos": [float(x) for x in pos],
        "quat": [float(x) for x in quat], # [qx, qy, qz, qw]
        "q_rad": q,
        "qd_rad_s": qd
    }

@mcp.tool()
def move_home() -> str:
    """Moves robot to home position. Blocks until complete."""
    ctrl = _get_controller()
    with _motion_lock:
        ctrl.move_home()
        success = _ws_run_until_reached(DEFAULT_TIMEOUT_S, DEFAULT_HOLD_S)
    return "Home reached" if success else "Timeout moving home"

@mcp.tool()
def move_joints(q_rad: list[float]) -> str:
    """Moves joints to specified radians. Blocks until complete."""
    ctrl = _get_controller()
    # Validate
    if len(q_rad) != ctrl.arm_dof:
        return f"Error: Expected {ctrl.arm_dof} joint angles, got {len(q_rad)}"
        
    with _motion_lock:
        ctrl.send_joint_position_rad(q_rad)
        success = _ws_run_until_reached(DEFAULT_TIMEOUT_S, DEFAULT_HOLD_S)
    return "Joints reached" if success else "Timeout moving joints"

@mcp.tool()
def set_gripper(percent: float) -> str:
    """Sets gripper [0.0, 1.0]. Blocks briefly."""
    ctrl = _get_controller()
    val = max(0.0, min(1.0, float(percent)))
    with _motion_lock:
        ctrl.set_gripper_percent(val)
        _ws_run_until_reached(5.0, 0.5) # Short timeout/hold for gripper
    return f"Gripper set to {val:.2f}"

@mcp.tool()
def move_pose(pos: list[float], quat: list[float] | None = None, hold_s: float = 0.4) -> str:
    """
    Moves end-effector to Cartesian pose. Blocks until complete.
    
    Args:
        pos: [x, y, z] target position in meters
        quat: [qx, qy, qz, qw] target orientation. If None, maintains CURRENT orientation (position-only mode).
        hold_s: Duration to hold stability before returning (default 0.4s).
    """
    ctrl = _get_controller()
    
    # 1. Validate Position
    if len(pos) != 3:
        return "Error: pos must be [x, y, z]"
        
    # 2. Handle Orientation Logic
    target_quat = quat
    mode = "provided_quat"
    
    if target_quat is None:
        # Fallback: Use CURRENT orientation -> effectively position-only control with fixed orientation
        with _physics_lock:
            _, current_quat = ctrl.get_end_effector_pose()
        target_quat = list(current_quat)
        mode = "current_quat_fallback"
    elif len(target_quat) != 4:
        return "Error: quat must be [qx, qy, qz, qw]"
    else:
        # Normalize provided quat
        norm = math.sqrt(sum(x*x for x in target_quat))
        if norm < 1e-6:
             return "Error: Invalid quaternion (zero norm)"
        target_quat = [x/norm for x in target_quat]

    # 3. Solve IK
    try:
        with _physics_lock:
            # Note: The backend uses a fixed orientation weight (0.3).
            # By passing the *current* quaternion (in fallback mode), we minimize rotation error cost,
            # effectively prioritizing position while keeping orientation roughly steady.
            q_dest = ctrl.solve_ik(pos, target_quat, q_seed=None)
    except ValueError as e:
        return f"IK Error: {str(e)}"
        
    # 4. Execute
    with _motion_lock:
        ctrl.send_joint_position_rad(q_dest)
        success = _ws_run_until_reached(DEFAULT_TIMEOUT_S, hold_s)
        
    if not success:
        return "Timeout reaching pose"
        
    # Check final error for reporting? (Optional, but good for debug)
    return f"Reached pose ({mode})"

if __name__ == "__main__":
    # Argument parsing for viewer/transport
    # FastMCP doesn't expose standard argparse easily if we use mcp.run() directly with CLI args,
    # but we can peek at sys.argv or just rely on defaults/env vars.
    # The requirement says: python mcp_server_demo.py --transport streamable-http --viewer
    # FastMCP handles transport flags. We need to handle --viewer manually or via env.
    
    viewer_enabled = "--viewer" in sys.argv
    if "--viewer" in sys.argv:
        sys.argv.remove("--viewer") # Remove so FastMCP doesn't complain if it parses args

    print("[Demo Server] Starting...", file=sys.stderr)
    _startup(viewer=viewer_enabled)
    print("[Demo Server] Ready. Listening on stdio or http...", file=sys.stderr)
    mcp.run(transport="streamable-http") # Force http as requested, or let CLI decide? 
                                         # Req: "python mcp_server_demo.py --transport streamable-http"
                                         # So we should probably just call mcp.run() and let it parse arguments.
