#!/usr/bin/env python3
"""
Sort Cubes — Autonomous sorting demo.

Connects to the Kinova MCP server, discovers all cubes in the scene,
and sorts them into the corresponding bins (red into red bin, blue into blue bin).

Usage:
    1. Start server:  KINOVA_SCENE=sorting_task.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
    2. Run sorter:    python kinova_middleware/sort_cubes.py
"""
from __future__ import annotations

import asyncio
import math
from fastmcp import Client

# ── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8000/mcp"

# Bin locations (XY world coords)
RED_BIN_X, RED_BIN_Y = 0.0, 0.35
BLUE_BIN_X, BLUE_BIN_Y = 0.0, -0.35

# The bins have a base that is 0.02m thick (size z is 0.01m).
BIN_BASE_Z = 0.02

# Clearances (metres)
PREGRASP_CLEARANCE = 0.10       # above cube top before descending
GRASP_OFFSET       = 0.008      # above cube top when grasping
LIFT_HEIGHT        = 0.25       # above cube top after grasping
PREPLACE_CLEARANCE = 0.15       # above placement z before descending
PLACE_OFFSET       = 0.04       # above target z when releasing
RETREAT_HEIGHT     = 0.20       # above placement z after releasing

# Stacking margin per layer (small gap to avoid physics collision pop)
STACK_MARGIN = 0.002

# Max stack height above table (workspace limit for this 4-DOF arm)
MAX_STACK_Z = 0.35

# Position-only quaternion — triggers position-only IK on the server
POS_ONLY_QUAT = [0.0, 0.0, 0.0, 0.0]

# Gripper percent (1.0 = fully open, 0.0 = fully closed)
GRIPPER_OPEN = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def pretty(result) -> dict:
    return result.structured_content or {}


async def get_full_q(client) -> list[float]:
    """Read the full joint state vector (for IK seeding)."""
    r = await client.call_tool("get_joint_state")
    return pretty(r).get("q_rad", [])


async def move_above(client, x: float, y: float, z: float, label: str = "", target_quat: list[float] | None = None):
    """Move EE to (x, y, z) using position-only IK with current q as seed, unless target_quat is provided."""
    q = await get_full_q(client)
    r = await client.call_tool("move_pose", {
        "target_pos": [x, y, z],
        "target_quat": target_quat if target_quat is not None else POS_ONLY_QUAT,
        "seed_q_rad": q,
        "allow_orientation_fallback": True,
    })
    data = pretty(r)
    status = data.get("status", "unknown")
    if status == "ok":
        log(f"✓ {label} OK", 2)
    elif status == "timeout":
        log(f"⚠ {label} TIMEOUT", 2)
    else:
        log(f"✗ {label} ERROR: {data.get('message')}", 2)
    return data


async def face_target(client, tx: float, ty: float):
    """Rotate robot base (joint_1) to face the target XY position."""
    BASE_X, BASE_Y = 0.0, 0.0
    dx = tx - BASE_X
    dy = ty - BASE_Y
    theta_world = math.atan2(dy, dx)
    # Convert to joint_1 frame (link_1 has 180° Y-rotation → negate)
    j1_target = -theta_world
    j1_target = math.atan2(math.sin(j1_target), math.cos(j1_target))

    r_joints = await client.call_tool("get_joint_state")
    arm_q = pretty(r_joints).get("q_rad", [])[:4]
    arm_q[0] = j1_target

    log(f"Facing target  θ={math.degrees(theta_world):.0f}°  → J1={math.degrees(j1_target):.0f}°", 2)
    r = await client.call_tool("move_joints", {"q": arm_q, "units": "rad"})
    data = pretty(r)
    if data.get("status") != "ok":
        log(f"⚠ Face TIMEOUT/ERROR", 2)
    return data


# ── Object Discovery ─────────────────────────────────────────────────────────

async def discover_cubes(client) -> list[dict]:
    """Query all objects, filter to cubes only."""
    r = await client.call_tool("get_object_pose", {"body_name": "all"})
    data = pretty(r)
    if data.get("status") == "error":
        log(f"✗ Object discovery failed: {data.get('message')}")
        return []

    all_objs = data.get("objects", [])
    cubes = []
    for obj in all_objs:
        name = obj.get("body_name", "")
        if "cube" in name.lower():
            cubes.append(obj)
    return cubes


def cube_half_height(obj: dict) -> float:
    """Get the half-height of a box geom (size[2] for MuJoCo box = half-extent in z)."""
    size = obj.get("size", [0.03, 0.03, 0.03])
    # MuJoCo box size = [half_x, half_y, half_z]
    return size[2] if len(size) >= 3 else 0.03


def cube_full_height(obj: dict) -> float:
    return 2.0 * cube_half_height(obj)


def cube_pos(obj: dict) -> tuple[float, float, float]:
    p = obj.get("position", {})
    return (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))


def planar_dist(obj: dict, dx: float, dy: float) -> float:
    x, y, _ = cube_pos(obj)
    return math.sqrt((x - dx) ** 2 + (y - dy) ** 2)


# ── Pick Protocol ────────────────────────────────────────────────────────────

async def align_wrist(client, obj: dict):
    """Align the EE wrist with the object's principal axis."""
    quat = obj.get("quaternion", {})
    obj_quat = [quat.get("qx", 0), quat.get("qy", 0),
                quat.get("qz", 0), quat.get("qw", 1)]

    r_ee = await client.call_tool("get_end_effector_pose")
    ee_q = pretty(r_ee).get("quaternion", {})
    ee_quat = [ee_q.get("qx", 0), ee_q.get("qy", 0),
               ee_q.get("qz", 0), ee_q.get("qw", 1)]

    r_align = await client.call_tool("compute_wrist_alignment", {
        "obj_quat_xyzw": obj_quat, "ee_quat_xyzw": ee_quat,
    })
    angle = pretty(r_align).get("angle_deg", 0.0)
    log(f"Aligning wrist by {angle:.1f}°", 2)
    r_rot = await client.call_tool("rotate_wrist", {"angle_deg": angle})
    data = pretty(r_rot)
    if data.get("status") != "ok":
        log(f"⚠ Wrist align: {data.get('message')}", 2)


async def pick_cube(client, obj: dict) -> tuple[bool, list[float]]:
    """Pick a cube from its current position. Returns (success_bool, grasped_quaternion)."""
    name = obj["body_name"]
    x, y, z = cube_pos(obj)
    hh = cube_half_height(obj)
    z_top = z + hh  # top surface of cube

    log(f"PICK {name} at ({x:.3f}, {y:.3f}, {z:.3f})  h={hh*2:.3f}m", 1)

    # Face the cube
    await face_target(client, x, y)

    # Open gripper
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})

    # Pregrasp: high above cube
    await move_above(client, x, y, z_top + PREGRASP_CLEARANCE, "Pregrasp")

    # Align wrist with cube orientation
    await align_wrist(client, obj)

    # Descend to grasp height
    await move_above(client, x, y, z_top + GRASP_OFFSET, "Descend")

    # Grasp
    log("Grasping at 55% …", 2)
    await client.call_tool("set_gripper", {"percent": 0.55})
    await asyncio.sleep(2.0)

    # Read current wrist orientation to maintain it perfectly vertical during lift and place
    r_ee_lift = await client.call_tool("get_end_effector_pose")
    ee_lift_q = pretty(r_ee_lift).get("quaternion", {})
    grasp_quat = [
        ee_lift_q.get("qx", 0), 
        ee_lift_q.get("qy", 0), 
        ee_lift_q.get("qz", 0), 
        ee_lift_q.get("qw", 1)
    ]

    # Lift maintaining exact wrist orientation
    await move_above(client, x, y, z_top + LIFT_HEIGHT, "Lift", target_quat=grasp_quat)

    # Grasp check: re-query cube position — if it moved up with EE, we have it
    await asyncio.sleep(0.3)
    r_check = await client.call_tool("get_object_pose", {"body_name": name})
    check_data = pretty(r_check)
    new_pos = check_data.get("position", {})
    new_z = new_pos.get("z", 0.0)

    if new_z > z + 0.02:  # cube lifted at least 2cm
        log(f"✓ Grasp confirmed: cube z {z:.3f} → {new_z:.3f}", 2)
        return True, grasp_quat

    log(f"✗ Grasp FAILED for {name} — skipping", 2)
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
    return False, []


# ── Place Protocol ───────────────────────────────────────────────────────────

async def drop_cube(client, bin_target_name: str, grasp_quat: list[float]) -> bool:
    """Drop the currently held cube directly over the specified bin target."""
    
    # 1. Fetch the drop target site location
    r_target = await client.call_tool("get_object_pose", {"body_name": bin_target_name})
    t_data = pretty(r_target)
    
    if t_data.get("status") == "error":
        log(f"✗ Could not find drop target {bin_target_name}: {t_data.get('message')}", 1)
        return False
        
    pos = t_data.get("position", {})
    dest_x, dest_y, dest_z = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.3)
    
    log(f"DROP over {bin_target_name} at ({dest_x:.3f}, {dest_y:.3f}, {dest_z:.3f})", 1)

    # 2. Face destination
    await face_target(client, dest_x, dest_y)

    # 3. Move directly to the target drop height above the bin (maintain grasped orientation)
    await move_above(client, dest_x, dest_y, dest_z, "Move over bin", target_quat=grasp_quat)

    # 4. Release (Drop)
    log("Opening gripper (Dropping cube) …", 2)
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
    await asyncio.sleep(0.5)

    log(f"✓ Dropped into {bin_target_name}", 2)
    return True


# ── Main Loop ────────────────────────────────────────────────────────────────

async def main():

    async with Client(SERVER_URL) as client:
        log("=" * 60)
        log("    SORT CUBES — Autonomous Sorting Demo")
        log("=" * 60)

        # 0. Home
        log("\n[Step 0] Homing arm …")
        await client.call_tool("move_home")
        await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
        log("✓ Home reached, gripper open.\n")

        # 1. Discover cubes
        log("[Step 1] Discovering cubes …")
        cubes = await discover_cubes(client)
        if not cubes:
            log("✗ No cubes found in scene. Exiting.")
            return
        
        red_cubes = [c for c in cubes if "red" in c['body_name'].lower()]
        blue_cubes = [c for c in cubes if "blue" in c['body_name'].lower()]

        log(f"Found {len(red_cubes)} red cube(s) and {len(blue_cubes)} blue cube(s):\n")

        # Sort each group by closest to robot origin (0, 0)
        red_cubes = sorted(red_cubes, key=lambda c: planar_dist(c, 0, 0))
        blue_cubes = sorted(blue_cubes, key=lambda c: planar_dist(c, 0, 0))
        
        # Determine overall turn order: we'll interleave or just process all red then all blue
        process_queue = red_cubes + blue_cubes

        # Track stacks
        red_stack_count = 0
        blue_stack_count = 0
        
        results = []

        # 4. Pick and place loop
        for i, cube_obj in enumerate(process_queue):
            name = cube_obj["body_name"]
            is_red = "red" in name.lower()
            ch = cube_full_height(cube_obj)
            
            dest_target = "red_bin_target" if is_red else "blue_bin_target"
            
            log(f"\n{'─'*50}")
            log(f"  Cube {i+1}/{len(process_queue)}: {name}  (going to {dest_target})")
            log(f"{'─'*50}")

            # Re-query
            r_fresh = await client.call_tool("get_object_pose", {"body_name": name})
            fresh = pretty(r_fresh)
            if fresh.get("status") == "error":
                log(f"⚠ {name} not found — skipping", 1)
                results.append((name, "missing"))
                continue

            cube_obj["position"] = fresh.get("position", cube_obj["position"])
            cube_obj["quaternion"] = fresh.get("quaternion", cube_obj["quaternion"])

            # Pick
            pick_ok, grasp_quat = await pick_cube(client, cube_obj)
            if not pick_ok:
                results.append((name, "pick_failed"))
                continue

            # Drop into bin
            place_ok = await drop_cube(client, dest_target, grasp_quat)
            if not place_ok:
                await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
                results.append((name, "place_failed"))
                continue

            results.append((name, "sorted"))

            log("Returning home …", 2)
            await client.call_tool("move_home")

        # Final home
        log(f"\n{'='*60}")
        log("  SORTING COMPLETE")
        log(f"{'='*60}\n")

        await client.call_tool("move_home")

        log(f"Red Bin: {red_stack_count} cubes. Blue Bin: {blue_stack_count} cubes.\n")
        log("Results:")
        for name, status in results:
            icon = "✓" if status == "sorted" else "✗"
            log(f"  {icon} {name:20s} → {status}")

        log(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
