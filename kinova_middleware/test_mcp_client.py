#!/usr/bin/env python3
"""
Comprehensive MCP client to test ALL Kinova MCP server tools.

Usage:
    1. Start the server:   .venv/bin/mjpython kinova_middleware/mcp_kinova_server.py
    2. In another terminal: python kinova_middleware/test_mcp_client.py
"""
import asyncio
import json
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


def pretty(result) -> str:
    """Pretty-print a CallToolResult."""
    if result.structured_content:
        return json.dumps(result.structured_content, indent=2)
    return str(result)


async def main():
    async with Client(SERVER_URL) as client:

        # ── 1. List tools ─────────────────────────────────────────────
        tools = await client.list_tools()
        print("=" * 60)
        print("  AVAILABLE TOOLS")
        print("=" * 60)
        for t in tools:
            print(f"  • {t.name}: {t.description[:80] if t.description else ''}")
        print()

        # ── 2. move_home ──────────────────────────────────────────────
        print("=" * 60)
        print("  TEST: move_home")
        print("=" * 60)
        result = await client.call_tool("move_home")
        print(pretty(result))
        print()

        # ── 3. get_end_effector_pose ──────────────────────────────────
        print("=" * 60)
        print("  TEST: get_end_effector_pose")
        print("=" * 60)
        result = await client.call_tool("get_end_effector_pose")
        print(pretty(result))
        print()

        # ── 4. get_joint_state ────────────────────────────────────────
        print("=" * 60)
        print("  TEST: get_joint_state")
        print("=" * 60)
        result = await client.call_tool("get_joint_state")
        print(pretty(result))
        print()

        # ── 5. set_gripper (open) ─────────────────────────────────────
        print("=" * 60)
        print("  TEST: set_gripper → OPEN (1.0)")
        print("=" * 60)
        result = await client.call_tool("set_gripper", {"percent": 1.0})
        print(pretty(result))
        print()

        # ── 6. set_gripper (close) ────────────────────────────────────
        print("=" * 60)
        print("  TEST: set_gripper → CLOSE (0.0)")
        print("=" * 60)
        result = await client.call_tool("set_gripper", {"percent": 0.0})
        print(pretty(result))
        print()

        # ── 7. move_joints (deg) ──────────────────────────────────────
        # Move all 4 arm joints to specific angles in degrees
        print("=" * 60)
        print("  TEST: move_joints (degrees)")
        print("=" * 60)
        target_joints_deg = [30.0, -45.0, 90.0, 10.0]
        print(f"  Target: {target_joints_deg} deg")
        result = await client.call_tool(
            "move_joints",
            {"q": target_joints_deg, "units": "deg"},
        )
        print(pretty(result))
        print()

        # ── 8. move_pose (position-only, P90 straight ahead) ─────────
        print("=" * 60)
        print("  TEST: move_pose → position-only (P90)")
        print("=" * 60)
        target_pos = [0.159, -0.244, 0.167]
        # Zero quaternion triggers position-only fallback
        target_quat = [0.0, 0.0, 0.0, 0.0]
        print(f"  Target pos: {target_pos}")
        result = await client.call_tool(
            "move_pose",
            {"target_pos": target_pos, "target_quat": target_quat},
        )
        print(pretty(result))
        print()

        # ── 9. move_pose (full pose with orientation) ─────────────────
        print("=" * 60)
        print("  TEST: move_pose → full pose with quaternion")
        print("=" * 60)
        # Move slightly up from P90 with a neutral-ish pointing-down orientation
        target_pos2 = [0.15, -0.20, 0.25]
        target_quat2 = [0.5, -0.5, -0.5, 0.5]  # roughly pointing down
        print(f"  Target pos:  {target_pos2}")
        print(f"  Target quat: {target_quat2}")
        result = await client.call_tool(
            "move_pose",
            {"target_pos": target_pos2, "target_quat": target_quat2},
        )
        print(pretty(result))
        print()

        # ── 10. Return home ───────────────────────────────────────────
        print("=" * 60)
        print("  TEST: move_home (return to start)")
        print("=" * 60)
        result = await client.call_tool("move_home")
        print(pretty(result))
        print()

        # ── 11. Final state check ─────────────────────────────────────
        print("=" * 60)
        print("  FINAL STATE")
        print("=" * 60)
        pose = await client.call_tool("get_end_effector_pose")
        joints = await client.call_tool("get_joint_state")
        print("Pose:")
        print(pretty(pose))
        print("Joints:")
        print(pretty(joints))
        print()

        print("✅ All tools tested!")


if __name__ == "__main__":
    asyncio.run(main())
