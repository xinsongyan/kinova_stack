import asyncio
import math
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

def pretty(result) -> dict:
    return result.structured_content or {}

async def test_reach_target():
    async with Client(SERVER_URL) as client:
        print("Connected to MCP Server.")
        
        # 0. Move Home
        print("Moving Home...")
        await client.call_tool("move_home")
        
        # 1. Get Target Pose
        target_name = "red_bin_target"
        print(f"Fetching pose for {target_name}...")
        r_target = await client.call_tool("get_object_pose", {"body_name": target_name})
        t_data = pretty(r_target)
        
        if t_data.get("status") == "error":
            print(f"Error finding target: {t_data.get('message')}")
            return
            
        pos = t_data.get("position", {})
        tx, ty, tz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
        print(f"Target {target_name} found at: X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}")
        
        # 2. Replicate face_target Logic
        BASE_X, BASE_Y = 0.0, 0.0
        dx = tx - BASE_X
        dy = ty - BASE_Y
        theta_world = math.atan2(dy, dx)
        j1_target = -theta_world
        j1_target = math.atan2(math.sin(j1_target), math.cos(j1_target))
        
        print(f"Facing Target -> rotating base J1 to {math.degrees(j1_target):.1f} degrees")
        r_joints = await client.call_tool("get_joint_state")
        arm_q = pretty(r_joints).get("q_rad", [])[:4]
        arm_q[0] = j1_target
        await client.call_tool("move_joints", {"q": arm_q, "units": "rad"})
        print("Faced Target.")
        
        # 3. Move Pose to Target Position
        print(f"Moving to Target using IK...")
        r_ee = await client.call_tool("get_end_effector_pose")
        ee_q = pretty(r_ee).get("quaternion", {})
        ee_quat = [ee_q.get("qx", 0), ee_q.get("qy", 0), ee_q.get("qz", 0), ee_q.get("qw", 1)]
        
        cur_q = pretty(await client.call_tool("get_joint_state")).get("q_rad", [])
        
        # We try to reach the position using the current orientation
        r_move = await client.call_tool("move_pose", {
            "target_pos": [tx, ty, tz],
            "target_quat": ee_quat,
            "seed_q_rad": cur_q,
            "allow_orientation_fallback": True
        })
        
        move_data = pretty(r_move)
        print(f"Move Result: {move_data.get('status')} - Pos Err: {move_data.get('pos_err', 0.0):.4f}m")
        if move_data.get("status") == "error":
            print(f"Error info: {move_data.get('message')}")

if __name__ == "__main__":
    asyncio.run(test_reach_target())
