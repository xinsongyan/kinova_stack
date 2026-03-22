#!/usr/bin/env python3
"""
OpenAI-powered cube stacking agent.

Uses GPT-4o with function calling to autonomously discover and stack
cubes by invoking MCP server tools.

Usage:
    1. Start server:  mjpython kinova_middleware/backend/mcp_kinova_server.py
    2. export OPENAI_API_KEY=sk-...
    3. python kinova_middleware/openai_stack_cubes.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from fastmcp import Client
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
MCP_URL = "http://127.0.0.1:8000/mcp"
MODEL   = "gpt-4o-mini"
MAX_TURNS = 60  # safety cap on agent iterations

SYSTEM_PROMPT = """\
You are a robotic arm controller operating a Kinova MICO 4-DOF arm in MuJoCo simulation.
Your goal: discover all cubes in the scene and stack them into a neat tower.

IMPORTANT CONSTRAINTS:
- The arm has 4 degrees of freedom. It cannot independently control full 6-DOF orientation.
- To use position-only IK (recommended), pass target_quat=[0,0,0,0] to move_pose.
  The server will automatically fall back to position-only IK.
- To maintain a specific wrist orientation (e.g. during lift/place with a grasped object),
  pass the quaternion you recorded from get_end_effector_pose.
- Gripper: 0.9 = open, 0.55-0.60 = closed for grasping cubes.
- After grasping, wait a moment, then verify the grasp by checking if the object's z increased.
- MuJoCo box size = [half_x, half_y, half_z]. The top of a cube at position z with size[2]=hz is at z+hz.
- Use compute_wrist_alignment + rotate_wrist to align fingers with the object before grasping.
- Stack destination: choose any clear area in the workspace (e.g. x=0.2, y=-0.2).
- Between cubes, return home with move_home.

WORKFLOW per cube:
1. get_object_pose to read fresh position
2. set_gripper to open
3. move_pose above the cube (position-only)
4. get_end_effector_pose + compute_wrist_alignment + rotate_wrist to align fingers
5. move_pose to descend to grasp height
6. set_gripper to close
7. Wait, then get_end_effector_pose to record grasp orientation
8. move_pose to lift (pass grasp quat to maintain orientation)
9. get_object_pose to verify grasp succeeded
10. move_pose to preplace, place, release, retreat
11. move_home

When done stacking all cubes, say DONE and stop calling tools.
"""

# ── OpenAI tool definitions (matching MCP server) ────────────────────────────
TOOLS = []


# ── Agent loop ───────────────────────────────────────────────────────────────

async def load_tools_from_mcp(mcp_client):
    """Fetch tools from MCP server and convert to OpenAI format."""
    print("Fetching tools from MCP server...")
    try:
        # FastMCP client's list_tools returns a list of Tool objects
        mcp_tools = await mcp_client.list_tools()
        openai_tools = []
        
        for tool in mcp_tools:
            # MCP Tool schema -> OpenAI Function schema
            function_def = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {}
            }
            openai_tools.append({
                "type": "function",
                "function": function_def
            })
            print(f" - Loaded tool: {tool.name}")
            
        return openai_tools
    except Exception as e:
        print(f"Failed to load tools: {e}")
        return []

async def run_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    openai = OpenAI(api_key=api_key)

    print("=" * 55)
    print("  OpenAI Cube Stacking Agent")
    print("=" * 55)
    print(f"  Model: {MODEL}")
    print(f"  MCP:   {MCP_URL}")
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": "Discover all cubes in the scene and stack them into a tower.",
    })


    async with Client(MCP_URL) as mcp:
        # Load tools dynamically
        global TOOLS
        TOOLS = await load_tools_from_mcp(mcp)
        
        if not TOOLS:
            print("Warning: No tools loaded from MCP server. Agent may fail.")
        
        for turn in range(MAX_TURNS):
            # ── Call OpenAI ──────────────────────────────────
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            messages.append(msg)

            # ── If the model wants to talk (no tool calls) ───
            if not msg.tool_calls:
                print(f"\n🤖 {msg.content}")
                if msg.content and "DONE" in msg.content.upper():
                    break
                # Give it a nudge if it stops early
                messages.append({
                    "role": "user",
                    "content": "Continue. If all cubes are stacked, say DONE.",
                })
                continue

            # ── Execute tool calls ───────────────────────────
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                print(f"  ⚡ {name}({json.dumps(args, separators=(',', ':'))})")

                try:
                    result = await mcp.call_tool(name, args)
                    result_data = result.structured_content or {}
                except Exception as e:
                    result_data = {"status": "error", "message": str(e)}

                result_json = json.dumps(result_data, default=str)
                # Truncate very long results for token efficiency
                if len(result_json) > 2000:
                    result_json = result_json[:2000] + "..."

                print(f"    → {result_json[:120]}{'…' if len(result_json) > 120 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

        else:
            print(f"\n⚠ Agent hit max turns ({MAX_TURNS}).")

    print("\n✓ Agent finished.")


if __name__ == "__main__":
    asyncio.run(run_agent())
