#!/usr/bin/env python3
"""
Unified FastMCP agent for Kinova task routing.

This client accepts a free-form user task, chooses one of the supported
workflows (`grab_shapes`, `sort_cubes`, or `stack_cubes`), fetches the
corresponding SOP prompt from the MCP server, and then executes the task
through MCP tool calls.

Usage:
  python kinova_middleware/llm_clients/ultimate_llm.py --task "sort the cubes"
  python kinova_middleware/llm_clients/ultimate_llm.py
"""

import argparse
import asyncio
import os
import sys

from fastmcp import Client
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from helper_functions import (
    CHECK_SORTING_STATUS_TOOL,
    CHECK_STACKING_STATUS_TOOL,
    FINISH_TASK_TOOL,
    VERIFY_OBJECT_LIFT_TOOL,
    bind_model_tools,
    build_retry_tool_message,
    check_sorting_progress,
    check_stacking_status,
    execute_mcp_tool,
    finish_task,
    invoke_with_rate_limit_retry,
    load_tools_and_prompts_from_mcp,
    parse_raw_tool_calls,
    verify_object_lift,
)


MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_AGENT_STEPS = 50

SYSTEM_PROMPT = """
You are a Kinova robot control agent that can handle exactly three workflows:
- `grab_shapes`
- `sort_cubes`
- `stack_cubes`

Routing rules:
1. Read the user's request and choose exactly one of the three workflows.
2. Before doing any task-specific action, fetch the matching SOP from the MCP server:
   - `get_prompt_grab_shapes`
   - `get_prompt_sort_cubes`
   - `get_prompt_stack_cubes(bottom_block='...', top_block='...')`
3. After fetching that prompt, follow its instructions closely and use the available tools accordingly.
4. Local helper tools are available when relevant:
   - `check_sorting_status` for sorting
   - `check_stacking_status` for cube stacking verification
   - `verify_object_lift` for grab/lift verification
   - `finish_task` to end the run when the task is done
5. If the user's request is ambiguous between workflows, ask one short clarification question instead of guessing.

Tool rules:
- Use only tools present in the tool schema.
- Never invent `pick_cube`, `place_cube`, or any other missing helper tools.
- Prefer structured tool calling over plain chat.
- Do not call more than one task prompt unless the user changes the task.
- Keep execution sequential and grounded in tool outputs.
"""


def build_local_tools() -> list[dict]:
    """Return local tool definitions exposed only inside this client."""
    return [CHECK_SORTING_STATUS_TOOL, CHECK_STACKING_STATUS_TOOL, VERIFY_OBJECT_LIFT_TOOL, FINISH_TASK_TOOL]


async def build_tools_schema(mcp_client) -> list[dict]:
    """Load MCP tools and prompts, then add local client-only helpers."""
    return await load_tools_and_prompts_from_mcp(
        mcp_client,
        extra_tools=build_local_tools(),
        skip_reset_scene=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Kinova LLM client.")
    parser.add_argument(
        "--task",
        default=None,
        help="Free-form user request. The agent routes it to grab_shapes, sort_cubes, or stack_cubes.",
    )
    args = parser.parse_args()
    task = args.task.strip() if args.task else input("Enter the task for the Kinova agent: ").strip()
    if not task:
        sys.exit("Error: task cannot be empty.")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("Error: NVIDIA_API_KEY not set in environment.")

    print(f"Initializing DeepSeek Agent ({MODEL_NAME}) via NVIDIA...")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
        base_url=BASE_URL,
    )

    print("Connecting to FastMCP Server at http://127.0.0.1:8000/mcp ...")
    try:
        async with Client("http://127.0.0.1:8000/mcp") as mcp_client:
            print("Connected! Fetching server capabilities...")
            tools_schema = await build_tools_schema(mcp_client)
            print(f"Successfully loaded {len(tools_schema)} callable tools (including prompt wrappers and local helpers).")

            llm_with_tools = bind_model_tools(llm, tools_schema, tool_choice="required")
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=task),
            ]

            iteration = 1
            while iteration <= MAX_AGENT_STEPS:
                print(f"--- [Thinking - Step {iteration}] ---")
                ai_msg = invoke_with_rate_limit_retry(llm_with_tools, messages)
                if ai_msg.content:
                    print(ai_msg.content)
                messages.append(ai_msg)

                tool_calls = list(ai_msg.tool_calls or [])
                if not tool_calls:
                    fallback_calls, fallback_tool_messages = parse_raw_tool_calls(ai_msg.content, iteration)
                    tool_calls.extend(fallback_calls)
                    messages.extend(fallback_tool_messages)
                    if fallback_tool_messages and not tool_calls:
                        tool_calls = [{"dummy": True}]

                if not tool_calls:
                    messages.append(
                        build_retry_tool_message(
                            iteration,
                            "No valid tool call was produced."
                        )
                    )
                    iteration += 1
                    continue

                for tool_call in tool_calls:
                    if "dummy" in tool_call:
                        continue
                    if tool_call["name"] == "finish_task":
                        result_str = await finish_task(mcp_client, tool_call["args"].get("summary", "Task finished."))
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=result_str,
                            )
                        )
                        print(f"\nFinal Report: {tool_call['args'].get('summary', 'Task finished.')}")
                        return
                    result_str = await execute_mcp_tool(
                        mcp_client,
                        tool_call["name"],
                        tool_call["args"],
                        local_tool_handlers={
                            "check_sorting_status": lambda client, tool_args: check_sorting_progress(client),
                            "check_stacking_status": lambda client, tool_args: check_stacking_status(client),
                            "verify_object_lift": lambda client, tool_args: verify_object_lift(
                                client,
                                tool_args.get("body_name", ""),
                                float(tool_args.get("min_height", 0.12)),
                            ),
                        },
                    )
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                            content=result_str,
                        )
                    )

                iteration += 1
            else:
                print(f"Safety Break: Reached the maximum of {MAX_AGENT_STEPS} iterations.")

    except Exception as exc:
        print(f"Execution failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
