#!/usr/bin/env python3
"""
Universal FastMCP Agent
=======================

This script acts as a completely agnostic, universal LLM agent that connects to ANY 
FastMCP server using `fastmcp.Client`. It has ZERO hardcoded logic about robot arms, 
shapes, or specific tools!

It uses LangChain to dynamically load whatever tools and Prompts the server happens 
to provide, and sets them up for `gpt-4o-mini` (or any model you choose).

Usage:
  python universal_fastmcp_agent.py --url "http://127.0.0.1:8000/mcp" --task "Clean up the blocks on the table"
"""

import asyncio
import os
import sys
import argparse
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from helper_functions import (
    FINISH_TASK_TOOL,
    VERIFY_OBJECT_LIFT_TOOL,
    build_retry_tool_message,
    execute_mcp_tool,
    finish_task,
    load_tools_and_prompts_from_mcp,
    parse_raw_tool_calls,
    verify_object_lift,
)

# =========================================================================
# CONFIGURATION
# -> Modify these variables to easily switch between models and APIs!
# =========================================================================
MODEL_NAME = "deepseek-ai/deepseek-v3.1"
BASE_URL = "https://integrate.api.nvidia.com/v1"


SYSTEM_PROMPT = """
You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.
Your mission is to sequentially pick up three objects in the scene: "box", "sphere", and "red_cylinder".

STANDARD OPERATING PROCEDURE FOR EACH OBJECT:
1. **Home & Prepare**: Call `move_home()`. Call `set_gripper(percent=1.0)` to open fingers.
2. **Locate Object**: Call `get_object_pose(body_name='<target>')`. Read its position, size, geom_type, and quaternion.
   - For box, geom_type="box". 
   - For sphere, geom_type="sphere". 
   - For red_cylinder, geom_type="cylinder".
3. **Compute Top Height**: Call `compute_grasp_height` using the object's geom_type, size, and quaternion to find `top_height`, the physical top boundary.
4. **Approach**: Call `move_pose` to the object's x,y, and Z = top_height + 0.15. Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
5. **Align Wrist**:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `compute_wrist_alignment` passing the object's quaternion and the EE quaternion.
     c. If box: the result angle might need modulus math to snap to a 90-degree face. Usually just apply the raw angle unless it's way off. (Hint: angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0)
     d. Call `rotate_wrist(angle_deg)` with the final alignment angle.
6. **Descend**: 
   Call `move_pose` to the object's x,y, and Z = top_height + 0.01.
   Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
7. **Grasp**: 
   Call `set_gripper(percent=...)`:
   - **sphere**: percent=0.62
   - **cylinder**: percent=0.55
   - **box**: percent=0.58
8. **Lift Safely**: Call `move_pose` using the current x,y, and Z = top_height + 0.20. 
   CRITICAL: Use `target_quat=[0.0, 0.0, 0.0, 0.0]` and pass the argument `"move_wrist": False` to prevent IK orientation failures while lifting straight up!
9. **Verify Lift**: After lifting, call the local tool `verify_object_lift(body_name='<target>')` to confirm the object's Z height is high enough.
   Use that verification result to decide whether the grasp succeeded before moving on.

Once you receive confirmation for all three objects (box, sphere, red_cylinder), state "All tasks complete."

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- When all tasks are complete, call `finish_task(summary=...)`.
- there is a z limit at 0.07 do not set the arm to go lower than this
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
                extra_tools=[VERIFY_OBJECT_LIFT_TOOL, FINISH_TASK_TOOL],
            )
                
            print(f"Successfully loaded {len(tools_schema)} callable tools (including local tools)!")
            
            # 3. Bind the extracted tools to our Langchain Agent
            llm_with_tools = llm.bind_tools([t["function"] for t in tools_schema])
            
            # Injecting a universal identity hint
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="Grab and lift the box, sphere, and red_cylinder. Start with the red_cylinder.")
            ]
            
            # Infinite Action Loop (stops when LLM gives up or finishes)
            iteration = 1
            while True:
                print(f"--- [Thinking - Step {iteration}] ---")
                
                # Predict next action
                ai_msg = llm_with_tools.invoke(messages)
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
                                "verify_object_lift": lambda client, tool_args: verify_object_lift(
                                    client,
                                    tool_args.get("body_name", ""),
                                    float(tool_args.get("min_height", 0.12)),
                                )
                            },
                            log_prefix="🚀 Agent Executing",
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

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
