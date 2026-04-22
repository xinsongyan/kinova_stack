#!/usr/bin/env python3
"""
Universal FastMCP Agent - Sorting Cubes
=======================

This script uses LangChain to connect to the FastMCP server and dynamically load 
tools to sort cubes into bins based on color.
"""

import asyncio
import os
import sys
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from helper_functions import (
    CHECK_SORTING_STATUS_TOOL,
    FINISH_TASK_TOOL,
    bind_model_tools,
    build_retry_tool_message,
    check_sorting_progress,
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
You are controlling a Kinova robot through MCP tools to perform a sorting task.
Your mission is to sort all red cubes into the red bin ("red_bin_target") and all blue cubes into the blue bin ("blue_bin_target").

CRITICAL TOOL RULES:
- **ONLY** use tools that are officially provided in the tool schema.
- **NEVER** invent or use a tool called `pick_cube` or `place_cube`. They DO NOT exist.
- You MUST execute the full 7-step pick sequence manually for every object.
- **ONLY** use the structured tool-calling format (no raw `<|tool_calls|>` text in the response).

Mission objective:
1. Use `check_sorting_status()` to identify the status of all cubes.
2. For each cube marked as "At Starting Position":
   a. Identify its color and target bin.
   b. Execute a precise 7-step pick-and-place sequence (see below).
3. Finish when `check_sorting_status()` reports "MISSION STATUS: ALL CUBES SORTED SUCCESSFULLY".

Detailed 7-Step Pick Sequence (REQUIRED):
1. `get_object_pose(body_name='...')` and `compute_grasp_height(...)`.
2. `move_pose` to `[x, y, top_height + 0.10]` with `target_quat=[0,0,0,0]` (Approach).
3. `get_end_effector_pose()` and `compute_wrist_alignment(...)` to find rotation.
4. `rotate_wrist(angle_deg)` to align the fingers.
5. `move_pose` to `[x, y, top_height + 0.015]` with `target_quat=[0,0,0,0]` (Descend).
6. `set_gripper(percent=0.54)` (Grasp).
7. `move_pose` to `[x, y, top_height + 0.20]` with `move_wrist=False` (Lift).

Safety and execution rules:
- NEVER attempt to pick up any cube marked as "OUT OF WORKSPACE".
- Focus strictly on cubes "At Starting Position".
- never go above 0.24m in the z axis when pregrasping.
- dont overthink ik error messages if the error is a couple of cm or less.

Output style:
- Briefly acknowledge the current state.
- Proceed to the NEXT tool call immediately.
- DO NOT summarize the whole task at once.
- When sorting is complete, call `finish_task(summary=...)`.
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
                extra_tools=[CHECK_SORTING_STATUS_TOOL, FINISH_TASK_TOOL],
                skip_reset_scene=True,
            )
                
            print(f"Successfully loaded {len(tools_schema)} callable tools (including local tools)!")
            
            # 4. Bind the extracted tools to our Langchain Agent
            llm_with_tools = bind_model_tools(llm, tools_schema, tool_choice="required")
            
            # Injecting a universal identity hint
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="Use check_sorting_status() then sort all cubes on the table.")
            ]
            
            # Infinite Action Loop (stops when LLM gives up or finishes)
            iteration = 1
            while iteration <= MAX_AGENT_STEPS:
                print(f"--- [Thinking - Step {iteration}] ---")
                
                # Predict next action
                ai_msg = llm_with_tools.invoke(messages)
                print(ai_msg.content)
                messages.append(ai_msg)
                
                # Check for tool calls (structured or raw text fallback)
                tool_calls = ai_msg.tool_calls or []
                
                # Fallback: Parse raw text tool calls if model outputs tags directly in content
                if not tool_calls and ai_msg.content and "<|tool▁call▁begin|>" in ai_msg.content:
                    parsed_tool_calls, parse_errors = parse_raw_tool_calls(ai_msg.content, iteration)
                    tool_calls.extend(parsed_tool_calls)
                    messages.extend(parse_errors)
                    if parse_errors and not tool_calls:
                        tool_calls = [{"dummy": True}]

                if tool_calls:
                    for tool_call in tool_calls:
                        if "dummy" in tool_call: continue # Skip the error marker
                        if tool_call["name"] == "finish_task":
                            result_str = await finish_task(mcp_client, tool_call["args"].get("summary", "Task finished."))
                            messages.append(
                                ToolMessage(
                                    tool_call_id=tool_call.get("id", f"c_{iteration}"),
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
                                "check_sorting_status": lambda client, tool_args: check_sorting_progress(client)
                            },
                        )
                        # Return the result back into the Agent's context
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call.get("id", f"c_{iteration}"), 
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
