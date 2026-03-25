#!/usr/bin/env python3
"""
Universal FastMCP Agent - Stacking Cubes
=======================

This script uses LangChain to connect to the FastMCP server and dynamically load 
tools to stack cubes, explicitly powered by the DeepSeek-v3.2 model via NVIDIA.
"""

import asyncio
import os
import sys
import json
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# =========================================================================
# CONFIGURATION
# =========================================================================
MODEL_NAME = "deepseek-ai/deepseek-v3.1"
BASE_URL = "https://integrate.api.nvidia.com/v1"

SYSTEM_PROMPT = """
You are controlling a Kinova robot through MCP tools. Your task is to stack the blue cube directly above the red cube.

Mission objective:
1. Identify the pose of the red cube and the blue cube.
2. Pick up the blue cube.
3. Place the blue cube centered on top of the red cube.
4. Finish safely.

Critical execution rule:
Before you close the gripper on the blue cube, you MUST calculate and store all pick-and-place target coordinates first, so the robot does not waste time while holding the cube and risk the cube slipping out.

Required planning order:
1. Query the blue cube pose.
2. Query the red cube pose.
3. Determine the blue cube size and red cube size if available.
4. Compute and mentally lock in these target positions BEFORE grasping:
   - blue pregrasp position
   - blue grasp position
   - blue lift position
   - target preplace position above red cube
   - target place position on top of red cube
   - target retreat position
5. Only after all of the above are known, execute the pick.
6. After grasping, move immediately through lift -> preplace -> place -> release -> retreat with no unnecessary pauses.

Tool-use constraints:
- Use only available MCP tools.
- Do not invent robot states or object poses.
- Always query real poses from tools.
- Do not delay after grasping except for the minimum needed to confirm the grasp.
- Do not recalculate placement while the cube is already in hand unless absolutely necessary.
- Keep motions efficient and direct.
- Prefer position-only IK when orientation is not essential, but align the wrist if needed for a stable grasp/place.
- If grasp verification fails, safely reopen the gripper, retreat, and try again logically.

Placement requirements:
- The blue cube must be centered over the red cube in x and y.
- The place height must correspond to the top surface of the red cube plus the blue cube’s half-height, with a small safe margin only if needed.
- The final stack should be vertical and stable.

Recommended execution sequence:
1. move_home
2. open gripper
3. get_object_pose(all) (this will return the pose of all objects)
4. compute all required pick/place coordinates in advance
5. Compute Top Height: Call `compute_grasp_height` using the object's geom_type, size, and quaternion to find `top_height`, the physical top boundary.
6. move to blue pregrasp Call `move_pose` to the object's x,y, and Z = top_height + 0.15. Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
7. Align Wrist:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `compute_wrist_alignment` passing the object's quaternion and the EE quaternion.
     c. If box: the result angle might need modulus math to snap to a 90-degree face. Usually just apply the raw angle unless it's way off. (Hint: angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0)
     d. Call `rotate_wrist(angle_deg)` with the final alignment angle.
8. Descend: 
   Call `move_pose` to the object's x,y, and Z = top_height + 0.01.
   Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
9. close gripper (0.58 is a good value for cubes)
10. immediately lift to precomputed lift point
11. move to precomputed preplace point above red cube
12. descend to precomputed place point
13. open gripper
14. retreat upward to precomputed retreat point
15. optionally verify final blue cube pose
16. return home

Reasoning rules:
- Think step by step.
- Be concise internally.
- Do not spend time over-planning after the grasp.
- The key requirement is: precompute lift, preplace, and place coordinates BEFORE picking up the blue cube.

Output style:
- Briefly state what you are doing before each major action.
- Report important pose values and errors when useful.
- If a step fails, explain why and recover safely.
"""

async def execute_mcp_tool(mcp_client, tool_name: str, args: dict) -> str:
    """Executes an MCP tool dynamically by name and returns a stringified JSON representation."""
    print(f"\n🚀 Agent Executing: {tool_name}({args})")
    
    # If the agent is trying to query a prompt (because we wrapped them as tools)
    if tool_name.startswith("get_prompt_"):
        prompt_name = tool_name.replace("get_prompt_", "", 1)
        try:
            result = await mcp_client.get_prompt(prompt_name, args)
            text = result.messages[0].content.text if hasattr(result.messages[0].content, "text") else str(result.messages[0].content)
            print(f"   → Read SOP Prompt: {prompt_name}")
            return text
        except Exception as e:
            return json.dumps({"error": f"Failed to get prompt: {str(e)}"})

    # Otherwise, it is a standard MCP tool call
    try:
        result = await mcp_client.call_tool(tool_name, args)
        result_json = json.dumps(result.structured_content or {}, default=str)
        # Prevent context explosion on giant return values
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "... [truncated]"
        print(f"   → Success: {result_json[:200]}...")
        return result_json
    except Exception as e:
        error_msg = json.dumps({"status": "error", "message": str(e)})
        print(f"   → ❌ Failed: {error_msg}")
        return error_msg

async def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key: sys.exit("Error: NVIDIA_API_KEY not set in environment.")
    print(f"Initializing DeepSeek Agent ({MODEL_NAME}) via NVIDIA...")
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        temperature=0, 
        api_key=api_key, 
        base_url=BASE_URL
    )

    print(f"Connecting to FastMCP Server at http://127.0.0.1:8000/mcp ...")
    try:
        async with Client("http://127.0.0.1:8000/mcp") as mcp_client:
            print("Connected! Fetching server capabilities...")
            
            # 1. Dynamically load native tools
            mcp_tools = await mcp_client.list_tools()
            tools_schema = []
            for tool in mcp_tools:
                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {}
                    }
                })
            
            # 2. Dynamically load server Prompts, wrap them as tools!
            mcp_prompts = await mcp_client.list_prompts()
            for prompt in mcp_prompts:
                properties = {}
                required = []
                if prompt.arguments:
                    for arg in prompt.arguments:
                        properties[arg.name] = {
                            "type": "string",
                            "description": arg.description or f"The {arg.name} to insert into the prompt"
                        }
                        if arg.required:
                            required.append(arg.name)

                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": f"get_prompt_{prompt.name}",
                        "description": prompt.description or f"Query the SOP/instructions for {prompt.name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                })
                
            print(f"Successfully loaded {len(mcp_tools)} tools and {len(mcp_prompts)} dynamic prompts!")
            
            # 3. Bind the extracted tools to our Langchain Agent
            llm_with_tools = llm.bind_tools([t["function"] for t in tools_schema])
            
            # Injecting a universal identity hint
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="Stack the blue_cube onto the red_cube.")
            ]
            
            # Infinite Action Loop (stops when LLM gives up or finishes)
            iteration = 1
            while True:
                print(f"--- [Thinking - Step {iteration}] ---")
                
                # Predict next action
                ai_msg = llm_with_tools.invoke(messages)
                print(ai_msg.content)
                messages.append(ai_msg)
                
                # Check if it wants to use tools
                if ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        result_str = await execute_mcp_tool(
                            mcp_client, 
                            tool_call["name"], 
                            tool_call["args"]
                        )
                        # Return the result back into the Agent's context
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"], 
                                name=tool_call["name"], 
                                content=result_str
                            )
                        )
                else:
                    # No tool calls, the AI just gave a conversational response (mission complete or clarifying question)
                    print(f"\n🤖 Agent Final Report: {ai_msg.content}")
                    break
                    
                iteration += 1

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
