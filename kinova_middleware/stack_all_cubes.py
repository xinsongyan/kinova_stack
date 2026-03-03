#!/usr/bin/env python3
"""
Stack All Cubes — Autonomous pick-and-place stacking demo.

Connects to the Kinova MCP server, discovers all cubes in the scene,
and stacks them into a neat vertical tower at a chosen destination.

Usage:
    1. Start server:  KINOVA_SCENE=02_multi_cubes.xml .venv/bin/mjpython kinova_middleware/backend/mcp_kinova_server.py
    2. Run stacker:   python kinova_middleware/stack_all_cubes.py
"""
from __future__ import annotations

import asyncio
import json
import math
from fastmcp import Client

# ── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8000/mcp"

# Destination tower centre (world XY) — clear area in front-right of robot
DEST_X, DEST_Y = 0.20, -0.20

# Clearances (metres)
PREGRASP_CLEARANCE = 0.08       # above cube top before descending
GRASP_OFFSET       = 0.008      # above cube top when grasping
LIFT_HEIGHT        = 0.3       # above cube top after grasping
PREPLACE_CLEARANCE = 0.1       # above placement z before descending
PLACE_OFFSET       = 0.04     # above target z when releasing
RETREAT_HEIGHT     = 0.10       # above placement z after releasing

# Stacking margin per layer (small gap to avoid physics collision pop)
STACK_MARGIN = 0.002

# Max stack height above table (workspace limit for this 4-DOF arm)
MAX_STACK_Z = 0.30

# Position-only quaternion — triggers position-only IK on the server
POS_ONLY_QUAT = [0.0, 0.0, 0.0, 0.0]

# Gripper percent (1.0 = fully open, 0.0 = fully closed)
GRIPPER_OPEN = 0.9


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
    pos_err = data.get("pos_err")
    err_str = f" pos_err={pos_err:.4f}m" if pos_err is not None else ""
    if status == "ok":
        log(f"✓ {label} OK{err_str}", 2)
    elif status == "timeout":
        log(f"⚠ {label} TIMEOUT{err_str}", 2)
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
    """Query all objects, filter to cubes/boxes only."""
    r = await client.call_tool("get_object_pose", {"body_name": "all"})
    data = pretty(r)
    if data.get("status") == "error":
        log(f"✗ Object discovery failed: {data.get('message')}")
        return []

    all_objs = data.get("objects", [])
    cubes = []
    for obj in all_objs:
        gtype = obj.get("geom_type", "")
        name = obj.get("body_name", "")
        # Accept "box" geom type, or names containing "cube" or "box"
        if gtype == "box" or "cube" in name.lower() or "box" in name.lower():
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

    # Simple position-based grasp
    log("Grasping at 60% …", 2)
    await client.call_tool("set_gripper", {"percent": 0.58})
    await asyncio.sleep(5.0) # Wait for fingers to close physically

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

    # Retry with slightly lower grasp
    log("⚠ Grasp check failed — retrying …", 2)
    await move_above(client, x, y, z_top + PREGRASP_CLEARANCE, "Re-pregrasp")
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})

    # Re-query position (cube may have shifted)
    r2 = await client.call_tool("get_object_pose", {"body_name": name})
    d2 = pretty(r2)
    p2 = d2.get("position", {})
    x2, y2, z2 = p2.get("x", x), p2.get("y", y), p2.get("z", z)
    z_top2 = z2 + hh

    await face_target(client, x2, y2)
    await move_above(client, x2, y2, z_top2 + PREGRASP_CLEARANCE, "Retry pregrasp")

    # Re-align wrist with fresh object orientation
    obj_retry = dict(obj, position=p2, quaternion=d2.get("quaternion", obj["quaternion"]))
    await align_wrist(client, obj_retry)

    await move_above(client, x2, y2, z_top2 + GRASP_OFFSET - 0.005, "Retry descend")

    # Retry grasp
    log("Retry Grasping at 60% …", 2)
    await client.call_tool("set_gripper", {"percent": 0.62})
    await asyncio.sleep(1.0)

    # Read orientation again for retry
    r_ee_lift2 = await client.call_tool("get_end_effector_pose")
    ee_lift_q2 = pretty(r_ee_lift2).get("quaternion", {})
    grasp_quat2 = [
        ee_lift_q2.get("qx", 0), 
        ee_lift_q2.get("qy", 0), 
        ee_lift_q2.get("qz", 0), 
        ee_lift_q2.get("qw", 1)
    ]

    await move_above(client, x2, y2, z_top2 + LIFT_HEIGHT, "Retry lift", target_quat=grasp_quat2)

    await asyncio.sleep(0.3)
    r3 = await client.call_tool("get_object_pose", {"body_name": name})
    d3 = pretty(r3)
    new_z2 = d3.get("position", {}).get("z", 0.0)
    if new_z2 > z2 + 0.02:
        log(f"✓ Retry grasp confirmed: cube z {z2:.3f} → {new_z2:.3f}", 2)
        return True, grasp_quat2

    log(f"✗ Grasp FAILED for {name} — skipping", 2)
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
    return False, []


# ── Place Protocol ───────────────────────────────────────────────────────────

async def place_cube(client, stack_index: int, cube_height: float, grasp_quat: list[float]) -> bool:
    """Place the currently held cube at the destination stack position, maintaining grasped orientation."""
    # Compute target z: table surface + half-cube + stacked layers
    # table_z is estimated during discovery; use global
    target_z = TABLE_Z + cube_height / 2 + (stack_index * (cube_height + STACK_MARGIN))

    if target_z > MAX_STACK_Z:
        log(f"⚠ Stack height {target_z:.3f}m exceeds max {MAX_STACK_Z}m — stopping", 1)
        return False

    log(f"PLACE at ({DEST_X:.3f}, {DEST_Y:.3f}, {target_z:.3f})  layer={stack_index}", 1)

    # Face destination
    await face_target(client, DEST_X, DEST_Y)

    # Align wrist before descending
    r_ee_lift = await client.call_tool("get_end_effector_pose")
    ee_lift_p = pretty(r_ee_lift).get("position", {})
    await move_above(client, ee_lift_p.get("x"), ee_lift_p.get("y"), ee_lift_p.get("z"), "Align wrist", target_quat=grasp_quat)

    # Preplace: high above target (pure position move)
    await move_above(client, DEST_X, DEST_Y, target_z + PREPLACE_CLEARANCE, "Preplace")

    # Descend to placement height (maintain grasped quat)
    await move_above(client, DEST_X, DEST_Y, target_z + PLACE_OFFSET, "Place descend", target_quat=grasp_quat)

    # Release
    log("Opening gripper …", 2)
    await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
    await asyncio.sleep(0.3)

    # Retreat upward (maintain grasped quat)
    await move_above(client, DEST_X, DEST_Y, target_z + RETREAT_HEIGHT, "Retreat", target_quat=grasp_quat)

    log(f"✓ Placed at layer {stack_index}", 2)
    return True


# ── Main Loop ────────────────────────────────────────────────────────────────

TABLE_Z = 0.0  # will be estimated from cube positions


async def main():
    global TABLE_Z

    async with Client(SERVER_URL) as client:
        log("=" * 60)
        log("    STACK ALL CUBES — Autonomous Stacking Demo")
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

        log(f"Found {len(cubes)} cube(s):")
        for c in cubes:
            x, y, z = cube_pos(c)
            h = cube_full_height(c)
            log(f"  • {c['body_name']:20s}  pos=({x:.3f}, {y:.3f}, {z:.3f})  h={h:.3f}m")

        # 2. Estimate table height
        TABLE_Z = min(cube_pos(c)[2] - cube_half_height(c) for c in cubes)
        log(f"\nEstimated table_z = {TABLE_Z:.4f}m")
        log(f"Destination = ({DEST_X:.3f}, {DEST_Y:.3f})")

        # 3. Sort cubes: closest to destination first
        cubes_sorted = sorted(cubes, key=lambda c: planar_dist(c, DEST_X, DEST_Y))
        log(f"\nStacking order (closest to dest first):")
        for i, c in enumerate(cubes_sorted):
            d = planar_dist(c, DEST_X, DEST_Y)
            log(f"  {i+1}. {c['body_name']}  (d={d:.3f}m)")

        # 4. Pick and place loop
        stack_index = 0
        results = []

        for i, cube_obj in enumerate(cubes_sorted):
            name = cube_obj["body_name"]
            ch = cube_full_height(cube_obj)
            log(f"\n{'─'*50}")
            log(f"  Cube {i+1}/{len(cubes_sorted)}: {name}  (stack layer {stack_index})")
            log(f"{'─'*50}")

            # Re-query position (cube may have shifted from earlier actions)
            r_fresh = await client.call_tool("get_object_pose", {"body_name": name})
            fresh = pretty(r_fresh)
            if fresh.get("status") == "error":
                log(f"⚠ {name} not found — skipping", 1)
                results.append((name, "missing"))
                continue

            # Update the object dict with fresh position
            cube_obj["position"] = fresh.get("position", cube_obj["position"])
            cube_obj["quaternion"] = fresh.get("quaternion", cube_obj["quaternion"])
            cube_obj["size"] = fresh.get("size", cube_obj["size"])

            # Pick
            pick_ok, grasp_quat = await pick_cube(client, cube_obj)
            if not pick_ok:
                results.append((name, "pick_failed"))
                continue

            # Place
            place_ok = await place_cube(client, stack_index, ch, grasp_quat)
            if not place_ok:
                # Drop the cube safely
                await client.call_tool("set_gripper", {"percent": GRIPPER_OPEN})
                results.append((name, "place_failed"))
                break  # height limit reached

            stack_index += 1
            results.append((name, "stacked"))

            # Quick home between cubes for a clean approach
            log("Returning home …", 2)
            await client.call_tool("move_home")

        # 5. Final home
        log(f"\n{'='*60}")
        log("  STACKING COMPLETE")
        log(f"{'='*60}\n")

        await client.call_tool("move_home")

        log(f"Tower at ({DEST_X:.3f}, {DEST_Y:.3f}), {stack_index} cube(s) stacked.\n")
        log("Results:")
        for name, status in results:
            icon = "✓" if status == "stacked" else "✗"
            log(f"  {icon} {name:20s} → {status}")

        log(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
