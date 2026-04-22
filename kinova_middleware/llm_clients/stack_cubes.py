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
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from helper_functions import (
    CHECK_STACKING_STATUS_TOOL,
    FINISH_TASK_TOOL,
    bind_model_tools,
    build_retry_tool_message,
    check_stacking_status,
    execute_mcp_tool,
    finish_task,
    load_tools_and_prompts_from_mcp,
    parse_raw_tool_calls,
)

# =========================================================================
# CONFIGURATION
# =========================================================================
MODEL_NAME = "deepseek-ai/deepseek-v3.1"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_AGENT_STEPS = 50

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
   Call `move_pose` to the object's x,y, and Z = top_height + 0.02.
   Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
9. close gripper (0.58 is a good value for cubes)
10. immediately lift to precomputed lift point
11. **Prelocate Stack Target**: Call `get_object_pose(body_name='red_cube')`. Note its position and half-height (size[2]).
12. **Precalculate Stacking Height**: Determine the center position for the blue cube: `target_z = red_z + red_hh + blue_hh + 0.01` (includes a small safety margin).
13. **Pre-Place**:
    a. Call `move_pose` to `[red_x, red_y, target_z + 0.15]` with `target_quat=[0.0, 0.0, 0.0, 0.0]`.
15. **Place Descend**: 
    Call `move_pose` to the corrected X,Y and `Z = target_z + 0.1`.
16. **Release and Retreat**: 
    a. Call `set_gripper(percent=1.0)`.
    b. Call `move_pose` straight up to `target_z + 0.15` with `"move_wrist": False`.
    c. Call `move_home()`.
17. **Verify Stack**:
    Call the local tool `check_stacking_status()` to confirm which cube is on top of which.
    Use that report to verify whether `blue_cube` ended up on `red_cube` or whether a different pair is stacked.

Reasoning rules:
- Think step by step.
- Be concise internally.
- Do not spend time over-planning after the grasp.
- The key requirement is: precompute lift, preplace, and place coordinates BEFORE picking up the blue cube.
- never go above 0.24m in the z axis

Output style:
- Briefly state what you are doing before each major action.
- Report important pose values and errors when useful.
- If a step fails, explain why and recover safely.
- When the task is complete, call `finish_task(summary=...)`.
"""

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
            tools_schema = await load_tools_and_prompts_from_mcp(
                mcp_client,
                extra_tools=[CHECK_STACKING_STATUS_TOOL, FINISH_TASK_TOOL],
                skip_reset_scene=True,
            )
                
            print(f"Successfully loaded {len(tools_schema)} callable tools (including local tools)!")
            
            # 3. Bind the extracted tools to our Langchain Agent
            llm_with_tools = bind_model_tools(llm, tools_schema, tool_choice="required")
            
            # Injecting a universal identity hint
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="Stack the blue_cube onto the red_cube.")
            ]
            
            # Infinite Action Loop (stops when LLM gives up or finishes)
            iteration = 1
            while iteration <= MAX_AGENT_STEPS:
                print(f"--- [Thinking - Step {iteration}] ---")
                
                # Predict next action
                ai_msg = llm_with_tools.invoke(messages)
                print(ai_msg.content)
                messages.append(ai_msg)
                
                tool_calls = list(ai_msg.tool_calls or [])
                if not tool_calls and ai_msg.content:
                    fallback_calls, fallback_tool_messages = parse_raw_tool_calls(ai_msg.content, iteration)
                    tool_calls.extend(fallback_calls)
                    messages.extend(fallback_tool_messages)
                    if fallback_tool_messages and not tool_calls:
                        tool_calls = [{"dummy": True}]

                # Check if it wants to use tools
                if tool_calls:
                    for tool_call in tool_calls:
                        if "dummy" in tool_call:
                            continue
                        if tool_call["name"] == "finish_task":
                            result_str = await finish_task(mcp_client, tool_call["args"].get("summary", "Task finished."))
                            messages.append(
                                ToolMessage(
                                    tool_call_id=tool_call["id"],
                                    name=tool_call["name"],
                                    content=result_str
                                )
                            )
                            print(f"\n🤖 Agent Final Report: {tool_call['args'].get('summary', 'Task finished.')}")
                            return
                        result_str = await execute_mcp_tool(
                            mcp_client, 
                            tool_call["name"], 
                            tool_call["args"],
                            local_tool_handlers={
                                "check_stacking_status": lambda client, tool_args: check_stacking_status(client)
                            },
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
                    messages.append(
                        build_retry_tool_message(
                            iteration,
                            "No valid tool call was produced."
                        )
                    )
                    
                iteration += 1
            else:
                print(f"Safety Break: Reached the maximum of {MAX_AGENT_STEPS} iterations.")

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
