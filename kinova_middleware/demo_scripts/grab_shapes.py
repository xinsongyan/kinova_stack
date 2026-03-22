#!/usr/bin/env python3
"""
Grab Shapes Demo — Explicitly grab different geometric primitives
using the math tools provided by the MCP server.

Targets: "box", "sphere", "red_cylinder" from the shapes.xml scene.

Usage:
    1. Start server:  mjpython kinova_middleware/backend/mcp_kinova_server.py
    (make sure to select shapes.xml scene)
    2. Run script:    python kinova_middleware/grab_shapes.py
"""
from __future__ import annotations

import asyncio
import math
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

POS_QUAT = [0.0, 0.0, 0.0, 0.0]  # triggers position-only IK
GRIP_OPEN = 1.0


def p(result) -> dict:
    return result.structured_content or {}


def log(msg: str, indent: int = 0):
    print(f"{'  ' * indent}{msg}", flush=True)


async def grab_object(client: Client, name: str, drop_x: float, drop_y: float):
    log(f"──────────────────────────────────────────────────")
    log(f"  Target: {name}")
    log(f"──────────────────────────────────────────────────")

    # 1. Discover object
    obj = p(await client.call_tool("get_object_pose", {"body_name": name}))
    if obj.get("status") == "error":
        log(f"  ✗ Could not find {name}")
        return

    pos = obj["position"]
    qx, qy, qz, qw = obj["quaternion"]["qx"], obj["quaternion"]["qy"], obj["quaternion"]["qz"], obj["quaternion"]["qw"]
    geom_type = obj.get("geom_type", "box")
    size = obj.get("size", [0.03, 0.03, 0.03])
    log(f"  Located at ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) - [Type: {geom_type}]", 1)

    # 2. Open Gripper
    await client.call_tool("set_gripper", {"percent": GRIP_OPEN})

    # 3. Calculate approach and grasp heights
    height_res = p(await client.call_tool("compute_grasp_height", {
        "geom_type": geom_type,
        "size": size,
        "quat_xyzw": [qx, qy, qz, qw]
    }))
    top_offset = height_res.get("top_height", 0.03)

    z_top = pos["z"] + top_offset
    approach_z = z_top + 0.15
    
    # ── STRATEGIES ──
    # Different objects need different gripper widths and Z-offsets
    if geom_type == "sphere":
        # Sphere: wider grip, don't descend all the way to midpoint
        grip_close = 0.62
        grasp_offset = 0.01
    elif geom_type == "cylinder":
        # Cylinder: tighter grip
        grip_close = 0.55
        grasp_offset = 0.008
    else:
        # Box
        grip_close = 0.58
        grasp_offset = 0.009
        
    grasp_z = z_top + grasp_offset

    # 4. Approach
    r = p(await client.call_tool("move_pose", {
        "target_pos": [pos["x"], pos["y"], approach_z],
        "target_quat": POS_QUAT,
    }))
    log(f"  Approach  err={r.get('pos_err', 0):.4f}m", 1)

    # 5. Orientation Strategy
    if geom_type in ["box", "cylinder"]:
        # Strategy: Must align wrist specifically to object's primary axes
        ee = p(await client.call_tool("get_end_effector_pose"))
        cur_q = ee.get("quaternion", {})
        ee_quat = [cur_q.get("qx",0), cur_q.get("qy",0), cur_q.get("qz",0), cur_q.get("qw",1)]
        
        align = p(await client.call_tool("compute_wrist_alignment", {
            "obj_quat_xyzw": [qx, qy, qz, qw],
            "ee_quat_xyzw": ee_quat
        }))
        angle_deg = align.get("angle_deg", 0.0)
        
        # Symmetries: boxes can be grabbed from 4 sides (90 deg symmetry)
        if geom_type == "box":
            angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0
            
        if abs(angle_deg) > 1.0:
            log(f"  Wrist aligned {angle_deg:.1f}° for {geom_type}.", 1)
            await client.call_tool("rotate_wrist", {"angle_deg": angle_deg})
    else:
        log(f"  Symmetric primitive ({geom_type}), skipping alignment.", 1)

    # 6. Descend
    r = p(await client.call_tool("move_pose", {
        "target_pos": [pos["x"], pos["y"], grasp_z],
        "target_quat": POS_QUAT,
    }))
    log(f"  Descend   err={r.get('pos_err', 0):.4f}m", 1)

    # 7. Grasp
    log(f"  Grasping ({grip_close*100:.0f}%) …", 1)
    await client.call_tool("set_gripper", {"percent": grip_close})
    await asyncio.sleep(0.5)
    
    # 8. Lift safely: freeze wrist and let positional IK handle the arm
    r = p(await client.call_tool("move_pose", {
        "target_pos": [pos["x"], pos["y"], approach_z],
        "target_quat": POS_QUAT,
        "move_wrist": False,
    }))
    log(f"  Lift      err={r.get('pos_err', 0):.4f}m", 1)

    # 9. Verify
    await asyncio.sleep(0.3)
    check = p(await client.call_tool("get_object_pose", {"body_name": name}))
    new_z = check.get("position", {}).get("z", 0)
    if new_z < pos["z"] + 0.05:
        log(f"  ✗ Grasp failed on {name} (z {pos['z']:.3f} → {new_z:.3f})", 1)
        await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
        await client.call_tool("move_home")
        return False
    else:
        log(f"  ✓ Valid grasp on {name} (z {pos['z']:.3f} → {new_z:.3f})", 1)
    
    await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
    log(f"  ✓ Grab and Dropped {name}")
    
    # 11. Return Home
    await client.call_tool("move_home")
    return True


async def main():
    async with Client(SERVER_URL) as client:
        log("=" * 55)
        log("  GRAB SHAPES")
        log("=" * 55)

        log("\n[0] Preparations …")
        await client.call_tool("move_home")
        await client.call_tool("reset_scene")
        
        # Grab Box (Cube)
        await grab_object(client, "box", drop_x=0.0, drop_y=-0.2)
        
        # Grab Sphere
        await grab_object(client, "sphere", drop_x=0.0, drop_y=0.2)
        
        # Grab Cylinder
        await grab_object(client, "red_cylinder", drop_x=0.2, drop_y=-0.2)
        
        log("\n[Done] All shapes picked up!")

if __name__ == "__main__":
    asyncio.run(main())
