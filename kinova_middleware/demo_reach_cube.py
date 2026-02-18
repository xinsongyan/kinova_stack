#!/usr/bin/env python3
"""
Demo: Find the red cube and move the arm to it.

Connects to the MCP Kinova server, queries the cube's position,
then commands the arm to reach that position using position-only IK.

Usage:
    1. Start server:  .venv/bin/mjpython kinova_middleware/mcp_kinova_server.py
    2. Run demo:      python kinova_middleware/demo_reach_cube.py
"""
import asyncio
import json
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

# Height offset so the gripper arrives just above the cube, not crashing into it
Z_OFFSET = 0.06  # metres above the cube centre

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


async def main():
    async with Client(SERVER_URL) as client:

        # 1. Home the arm
        print("=" * 50)
        print("  Step 1: Homing arm …")
        print("=" * 50)
        r = await client.call_tool("move_home")
        print(pretty(r))
        print()

        # 2. Open gripper
        print("=" * 50)
        print("  Step 2: Opening gripper …")
        print("=" * 50)
        r = await client.call_tool("set_gripper", {"percent": 0.9})
        print(pretty(r))
        print()

        # 3. Get the cube's position
        print("=" * 50)
        print("  Step 3: Querying cube position …")
        print("=" * 50)
        r = await client.call_tool("get_object_pose", {"body_name": "cube"})
        cube_data = r.structured_content or {}
        print(pretty(r))
        print()

        cube_pos = cube_data.get("position", {})
        cx = cube_pos.get("x", 0)
        cy = cube_pos.get("y", 0)
        cz = cube_pos.get("z", 0)
        print(f"  Cube at: x={cx:.4f}  y={cy:.4f}  z={cz:.4f}")

        # 4. Get current EE pose for reference
        print()
        print("=" * 50)
        print("  Step 4: Reading current EE pose …")
        print("=" * 50)
        r = await client.call_tool("get_end_effector_pose")
        ee_data = r.structured_content or {}
        print(pretty(r))
        print()

        # 5. Move above the cube (approach from above)
        approach_pos = [cx, cy, cz + Z_OFFSET + 0.08]
        print("=" * 50)
        print(f"  Step 5: Approaching above cube → {[f'{v:.4f}' for v in approach_pos]}")
        print("=" * 50)
        r = await client.call_tool("move_pose", {
            "target_pos": approach_pos,
            "target_quat": POS_ONLY_QUAT,
        })
        print(pretty(r))
        print()

        # 5b. Rotate wrist to align with rectangle
        print("=" * 50)
        print("  Step 5b: Rotating wrist to align …")
        print("=" * 50)
        r = await client.call_tool("rotate_wrist", {"angle_deg": -90.0})
        print(pretty(r))
        print()
        check_result(r, "approach")

        # 6. Descend to the cube
        target_pos = [cx, cy, cz + Z_OFFSET]
        print("=" * 50)
        print(f"  Step 6: Descending to cube → {[f'{v:.4f}' for v in target_pos]}")
        print("=" * 50)
        r = await client.call_tool("move_pose", {
            "target_pos": target_pos,
            "target_quat": POS_ONLY_QUAT,
        })
        check_result(r, "descend")

        # 7. Close gripper (grab)
        print("=" * 50)
        print("  Step 7: Closing gripper …")
        print("=" * 50)
        r = await client.call_tool("set_gripper", {"percent": 0.55})
        print(pretty(r))
        print()

        # 8. Lift up
        lift_pos = [cx, cy, cz + Z_OFFSET + 0.15]
        print("=" * 50)
        print(f"  Step 8: Lifting → {[f'{v:.4f}' for v in lift_pos]}")
        print("=" * 50)
        r = await client.call_tool("move_pose", {
            "target_pos": lift_pos,
            "target_quat": POS_ONLY_QUAT,
        })
        check_result(r, "lift")

        # 9. Check where the cube ended up
        print("=" * 50)
        print("  Step 9: Final cube position:")
        print("=" * 50)
        r = await client.call_tool("get_object_pose", {"body_name": "cube"})
        print(pretty(r))
        print()

        # 10. Final EE pose
        print("=" * 50)
        print("  Step 10: Final EE pose:")
        print("=" * 50)
        r = await client.call_tool("get_end_effector_pose")
        print(pretty(r))
        print()

        print("✅ Reach-cube demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
