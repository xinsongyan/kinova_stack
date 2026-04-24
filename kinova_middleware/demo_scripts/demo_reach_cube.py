#!/usr/bin/env python3
"""
Demo: Find the red cube and move the arm to it.

Connects to the MCP Kinova server, queries the cube's position,
then commands the arm to reach that position using position-only IK.

Usage:
    1. Start server:  .venv/bin/mjpython kinova_middleware/backend/mcp_server/app.py
    2. Run demo:      python kinova_middleware/demo_reach_cube.py
"""
import asyncio
import json
from fastmcp import Client
from time import sleep

SERVER_URL = "http://127.0.0.1:8000/mcp"

# Small clearance above the computed grasp centre (metres)
GRASP_CLEARANCE = 0.005

# Position-only quaternion — triggers position-only IK on the server
POS_ONLY_QUAT = [0.0, 0.0, 0.0, 0.0]


def pretty(result) -> str:
    if result.structured_content:
        return json.dumps(result.structured_content, indent=2)
    return str(result)


def check_result(result, step_name: str):
    """Print result and warn if timeout/error."""
    data = result.structured_content or {}
    status = data.get("status", "unknown")
    pos_err = data.get("pos_err")
    print(pretty(result))
    if status == "error":
        print(f"  ⚠ ERROR in {step_name}: {data.get('message')}")
    elif status == "timeout":
        err_str = f"pos_err={pos_err:.4f}m" if pos_err is not None else ""
        print(f"  ⚠ TIMEOUT in {step_name} ({err_str})")
    else:
        err_str = f"pos_err={pos_err:.4f}m" if pos_err is not None else ""
        print(f"  ✓ {step_name} OK ({err_str})")
    print()
    return data


async def grab_object(client, body_name: str):
    print(f"\n{'='*60}")
    print(f"  STARTING SEQUENCE FOR: {body_name}")
    print(f"{'='*60}\n")

    # 1. Home the arm
    print("  Step 1: Homing arm …")
    await client.call_tool("move_home")

    # 2. Open gripper
    print("  Step 2: Opening gripper …")
    await client.call_tool("set_gripper", {"percent": 0.9})

    # 3. Get Pose & Compute Grasp Height
    print(f"  Step 3: Querying {body_name} pose …")
    r = await client.call_tool("get_object_pose", {"body_name": body_name})
    data = r.structured_content or {}
    if data.get("status") == "error":
        print(f"  ⚠ Skipped {body_name}: {data.get('message')}")
        return

    pos = data.get("position", {})
    cx, cy, cz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
    geom_type = data.get("geom_type", "unknown")
    size = data.get("size", [])
    quat = data.get("quaternion", {})
    quat_xyzw = [quat.get("qx", 0), quat.get("qy", 0), quat.get("qz", 0), quat.get("qw", 1)]

    r_height = await client.call_tool("compute_grasp_height", {
        "geom_type": geom_type, "size": size, "quat_xyzw": quat_xyzw
    })
    h_data = r_height.structured_content or {}
    top_height = h_data.get("top_height", 0.0)
    grasp_z = top_height + GRASP_CLEARANCE

    print(f"  -> Found {body_name} at ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"  -> Grasp z-offset: {grasp_z:.4f}m")

    # 3b. Face the Object (Critical for reachability)
    import math

    # Robot base position in world XY (from MJCF: base is at origin)
    BASE_X, BASE_Y = 0.0, 0.0

    # Direction vector from base to object (projected onto XY plane)
    dx = cx - BASE_X
    dy = cy - BASE_Y

    # World-frame yaw angle to the object
    theta_world = math.atan2(dy, dx)

    # Convert to joint_1 frame
    joint_1_target = -theta_world

    # Wrap to [-π, π] to stay within joint limits
    joint_1_target = math.atan2(math.sin(joint_1_target), math.cos(joint_1_target))

    print(f"  Step 3b: Facing object  θ_world={math.degrees(theta_world):.1f}°  →  J1={math.degrees(joint_1_target):.1f}°")

    # Get current joints (Home)
    r_joints = await client.call_tool("get_joint_state")
    full_q = r_joints.structured_content.get("q_rad", [])

    # Only the first 4 values are arm joints; the rest are fingers.
    arm_q = full_q[:4]
    arm_q[0] = joint_1_target

    # Move joints (rotate base to face object)
    r_face = await client.call_tool("move_joints", {"q": arm_q, "units": "rad"})
    check_result(r_face, "Face Object")

    # 4. Approach
    approach_pos = [cx, cy, cz + grasp_z + 0.15] # 15cm approach
    print(f"  Step 4: Approaching {approach_pos}")
    
    # Get current joints to seed the IK (keeps elbow up)
    r_joints = await client.call_tool("get_joint_state")
    
    # FIX: "Expected at least 10 seed joints, got 4"
    # The IK solver needs the FULL state vector (including fingers) as seed.
    full_q = r_joints.structured_content.get("q_rad", [])
    
    r_approach = await client.call_tool("move_pose", {
        "target_pos": approach_pos, 
        "target_quat": POS_ONLY_QUAT,
        "seed_q_rad": full_q 
    })
    check_result(r_approach, "Approach")

    # 5. Align Wrist
    print("  Step 5: Aligning wrist …")
    r_ee = await client.call_tool("get_end_effector_pose")
    ee_q = r_ee.structured_content.get("quaternion", {})
    ee_quat = [ee_q.get("qx", 0), ee_q.get("qy", 0), ee_q.get("qz", 0), ee_q.get("qw", 1)]
    
    r_align = await client.call_tool("compute_wrist_alignment", {
        "obj_quat_xyzw": quat_xyzw, "ee_quat_xyzw": ee_quat
    })
    wrist_angle = r_align.structured_content.get("angle_deg", 0.0)
    print(f"    -> Rotating wrist by {wrist_angle:.2f} deg")
    r_wrist = await client.call_tool("rotate_wrist", {"angle_deg": wrist_angle})
    check_result(r_wrist, "Align Wrist")

    # 6. Descend
    target_pos = [cx, cy, cz + grasp_z]
    print(f"  Step 6: Descending to {target_pos} …")
    
    # Get current joints (now with aligned wrist) to seed IK
    r_joints = await client.call_tool("get_joint_state")
    current_q = r_joints.structured_content.get("q_rad", [])
    
    r_descend = await client.call_tool("move_pose", {
        "target_pos": target_pos, 
        "target_quat": POS_ONLY_QUAT,
        "seed_q_rad": current_q
    })
    check_result(r_descend, "Descend")

    # 7. Grab
    print("  Step 7: Grasping …")
    await client.call_tool("set_gripper", {"percent": 0.6})

    # 8. Lift
    lift_pos = [cx, cy, cz + grasp_z + 0.20]
    print("  Step 8: Lifting …")
    
    # Get current joints (now with closed gripper) to seed IK
    r_joints = await client.call_tool("get_joint_state")
    current_q = r_joints.structured_content.get("q_rad", [])
    
    # Get current wrist orientation to maintain it perfectly vertical during lift
    r_ee_lift = await client.call_tool("get_end_effector_pose")
    ee_lift_q = r_ee_lift.structured_content.get("quaternion", {})
    lift_quat = [
        ee_lift_q.get("qx", 0), 
        ee_lift_q.get("qy", 0), 
        ee_lift_q.get("qz", 0), 
        ee_lift_q.get("qw", 1)
    ]
    
    r_lift = await client.call_tool("move_pose", {
        "target_pos": lift_pos, 
        "target_quat": lift_quat, # Maintain exact grasp orientation
        "seed_q_rad": current_q
    })
    check_result(r_lift, "Lift")
    sleep(2) # Give viewer a moment to show the success before dropping

    # 9. Drop
    print("  Step 9: Dropping object …")
    await client.call_tool("set_gripper", {"percent": 0.9})

    print(f"  ✓ {body_name} sequence complete.")


def check_result(result, step_name: str):
    """Helper to verify tool result and print status."""
    content = result.structured_content or {}
    status = content.get("status", "unknown")
    if status != "ok":
        print(f"  ⚠ {step_name} FAILED: {content.get('message')}")
    else:
        # Optional: Print error metrics if available
        pos_err = content.get("pos_err")
        rot_err = content.get("rot_err")
        err_str = []
        if pos_err is not None: err_str.append(f"pos_err={pos_err:.4f}m")
        if rot_err is not None: err_str.append(f"rot_err={rot_err:.4f}rad")
        if err_str:
            print(f"    ✓ {step_name} OK ({', '.join(err_str)})")
        else:
            print(f"    ✓ {step_name} OK")


async def main():
    async with Client(SERVER_URL) as client:
        # 0. Discover Objects
        print("=" * 50)
        print("  Step 0: Discovering objects …")
        print("=" * 50)
        r = await client.call_tool("get_object_pose", {"body_name": "all"})
        data = r.structured_content or {}
        if data.get("status") == "error":
             print(f"Error discovering objects: {data.get('message')}")
             return
             
        objects_list = data.get("objects", [])
        print(f"  Found {len(objects_list)} movable objects:")
        for obj in objects_list:
            print(f"    - {obj['body_name']} ({obj['geom_type']}): size={obj['size']} pos={obj['position']} quat={obj['quaternion']}")
        
        # Sort objects by some criteria? Or just iterate.
        # Let's just iterate through the names.
        target_names = [obj['body_name'] for obj in objects_list]
        
        # Filter if needed (e.g. exclude some known non-targets if any appear). 
        # But our filter in server (freejoint) should be good.

        for body_name in target_names:
            await grab_object(client, body_name)
            
            # Drop/Reset
            print("  -> Dropping object …")
            await client.call_tool("set_gripper", {"percent": 0.9})
            await asyncio.sleep(1.0) # wait for drop

        print("\n✅ All discovered objects handled!")
        # Final home
        await client.call_tool("move_home")


if __name__ == "__main__":
    asyncio.run(main())
