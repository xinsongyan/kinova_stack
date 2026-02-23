"""
OpenAI-driven Kinova Arm Controller
===================================
This script uses the OpenAI API to control the Kinova Gen3 arm via the MCP server.
It provides the LLM with a system prompt detailing the "Standard Operating Procedure"
for picking up the cylinder, derived from the `demo_reach_cube.py` logic.

Prerequisites:
    - MCP Server running: .venv/bin/mjpython kinova_middleware/backend/mcp_kinova_server.py
    - OPENAI_API_KEY environment variable set.
"""

import asyncio
import os
import sys
import json
from openai import OpenAI
from fastmcp import Client

# Configuration
SERVER_URL = "http://127.0.0.1:8000/mcp"
MODEL_NAME = "gpt-4o-mini"  # Use a capable model

# Tool Definitions will be loaded dynamically
TOOLS = []

SYSTEM_PROMPT = """
You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.
Your goal is to pick up the "red_cylinder" (blue box and green sphere are also present).

STANDARD OPERATING PROCEDURE:
1. **Home**: Call `move_home()` to reset the arm.
2. **Open Gripper**: Call `set_gripper(percent=0.9)` to open fingers.
3. **Plan Grasp**: Call `plan_grasp(body_name='red_cylinder')`.
   - This tool returns `approach_pose`, `grasp_pose`, and `wrist_angle_deg`.
4. **Approach**: Call `move_pose` using the `approach_pose` from step 3.
   - Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
5. **Align Wrist**: Call `rotate_wrist` using `wrist_angle_deg` from step 3.
6. **Descend**: Call `move_pose` using the `grasp_pose` from step 3.
   - Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
7. **Grasp**: Call `set_gripper(percent=0.5)` to close fingers.
8. **Lift**: Call `move_pose` using the current x,y but z + 0.20.

Output policy:
- Prefer tool calls over chat.
- Do not invent values; compute from tool results.
"""

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

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Initializing OpenAI client...")
    openai_client = OpenAI(api_key=api_key)

    print(f"Connecting to MCP Server at {SERVER_URL}...")
    try:
        async with Client(SERVER_URL) as mcp_client:
            print("Connected to MCP Server.")
            
            # Load tools dynamically
            global TOOLS
            TOOLS = await load_tools_from_mcp(mcp_client)
            
            if not TOOLS:
                print("Warning: No tools loaded from MCP server. Agent may fail.")

            # Initial message
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Begin the sequence to grab the cylinder."}
            ]

            print("\n--- Starting Mission ---\n")

            while True:
                # Call OpenAI
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

                message = response.choices[0].message
                messages.append(message)

                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        args_str = tool_call.function.arguments
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            print(f"Error decoding arguments: {args_str}")
                            continue

                        print(f"🤖 AI Calling Tool: {func_name}({args})")
                        
                        # Execute tool on MCP
                        content_str = ""
                        try:
                            result = await mcp_client.call_tool(func_name, args)
                            # Inspect the result structure
                            # FastMCP client returns ToolResult which has `content` list
                            if hasattr(result, 'content') and result.content:
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        content_str += item.text
                                    else:
                                        content_str += str(item)
                            else:
                                content_str = str(result)
                            
                            # Log success (truncate for readability)
                            log_msg = content_str.replace('\n', ' ')
                            print(f"   → Result: {log_msg}")

                        except Exception as e:
                            content_str = f"Error executing tool: {str(e)}"
                            print(f"   → Error: {content_str}")

                        # Feed result back to OpenAI
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content_str,
                        })
                else:
                    # No tool call, just text response
                    print(f"\n🤖 AI Message: {message.content}\n")
                    
                    # Heuristic for completion
                    if "lift" in message.content.lower() and ("complete" in message.content.lower() or "done" in message.content.lower()):
                        print("Mission likely complete. Exiting.")
                        break
                    
                    # If AI just talks, prompt it to continue if strictly necessary, 
                    # but usually it will just tell you it's done.
                    if "complete" in message.content.lower():
                        break

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
