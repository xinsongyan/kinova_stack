#!/usr/bin/env python3
"""
Dedicated stack_cubes LLM client.

This client uses the same routing prompt as `ultimate_llm.py`, but it is
preconfigured for the `stack_cubes` workflow and includes the local stacking
status helper for verification.

Examples:
  python kinova_middleware/llm_clients/stack_cubes.py
  python kinova_middleware/llm_clients/stack_cubes.py --model openai/gpt-oss-120b
"""

import argparse
import asyncio
import json
import os
import sys

from fastmcp import Client
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from helper_functions import (
    CHECK_STACKING_STATUS_TOOL,
    FINISH_TASK_TOOL,
    bind_model_tools,
    build_action_reference,
    build_reasoned_action_tool,
    build_retry_tool_message,
    check_stacking_status,
    execute_mcp_tool,
    finish_task,
    format_action_result,
    invoke_with_rate_limit_retry,
    load_tools_and_prompts_from_mcp,
    parse_raw_tool_calls,
    reset_scene_if_available,
)


DEFAULT_MODEL_NAME = "moonshotai/kimi-k2-instruct-0905"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_SCENE_NAME = "multi_cubes.xml"
DEFAULT_TASK = "Stack the green cube onto the red cube."
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
   - `check_stacking_status` for stacking verification
   - `finish_task` to end the run when the task is done
5. If the user's request is ambiguous between workflows, ask one short clarification question instead of guessing.

Tool rules:
- Use only tools present in the action reference.
- Never invent `pick_cube`, `place_cube`, or any other missing helper tools.
- Use the single structured wrapper tool `call_tool_with_reason` for every action.
- For every action, provide:
  - `reason`: one short sentence explaining what you are about to do and why
  - `tool_name`: the exact action name from the action reference
  - `tool_args`: a JSON object containing the arguments for that action
- Example:
  - `reason`: "I am going to check the current stack state so I know whether a placement is needed."
  - `tool_name`: `check_stacking_status`
  - `tool_args`: `{}`
- Exactly one `call_tool_with_reason` call is allowed per assistant turn.
- Do not bundle multiple actions into one response. Wait for the tool result before deciding the next action.
- Do not call more than one task prompt unless the user changes the task.
- Keep execution sequential and grounded in tool outputs.

Completion rules for `stack_cubes`:
- Use `check_stacking_status()` after placement attempts and before claiming success.
- Do not claim the stacking task is complete unless the status report confirms the intended stack.
- Call `finish_task(summary=...)` only after you have finished the task or cannot continue safely.
"""


def build_local_tools() -> list[dict]:
    """Return client-only helper tools for the stack_cubes workflow."""
    return [CHECK_STACKING_STATUS_TOOL, FINISH_TASK_TOOL]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Dedicated stack_cubes LLM client.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task instruction for the model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name to run.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="FastMCP server URL.")
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS, help="Maximum agent iterations.")
    parser.add_argument(
        "--scene-number",
        type=int,
        default=None,
        help=f"Optional 1-based scene number to load before the run. Defaults to {DEFAULT_SCENE_NAME}.",
    )
    parser.add_argument(
        "--tool-choice",
        default="required",
        choices=["required", "auto", "none", "default"],
        help="Tool-choice mode passed to the OpenAI-compatible model API.",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        sys.exit("Error: --max-steps must be at least 1.")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("Error: NVIDIA_API_KEY not set in environment.")

    print(f"Initializing stack_cubes agent ({args.model}) via NVIDIA...")
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        api_key=api_key,
        base_url=args.base_url,
    )

    print(f"Connecting to FastMCP Server at {args.server_url} ...")
    try:
        async with Client(args.server_url) as mcp_client:
            available_tool_names = {tool.name for tool in await mcp_client.list_tools()}
            print(f"Connected! Server exposes {len(available_tool_names)} tools.")
            await reset_scene_if_available(
                mcp_client,
                available_tool_names,
                scene_name=DEFAULT_SCENE_NAME,
                scene_number=args.scene_number,
                total_runs=1,
            )
            print("Fetching server capabilities...")
            tools_schema = await load_tools_and_prompts_from_mcp(
                mcp_client,
                extra_tools=build_local_tools(),
                skip_reset_scene=True,
            )

            print(f"Successfully loaded {len(tools_schema)} callable tools (including prompt wrappers and local helpers).")
            action_reference = build_action_reference(tools_schema)
            llm_with_tools = bind_model_tools(
                llm,
                [build_reasoned_action_tool(tools_schema)],
                tool_choice=args.tool_choice,
                parallel_tool_calls=False,
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=action_reference),
                HumanMessage(content=args.task),
            ]

            iteration = 1
            while iteration <= args.max_steps:
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
                            "No valid tool call was produced.",
                        )
                    )
                    iteration += 1
                    continue

                actionable_tool_calls = [tool_call for tool_call in tool_calls if "dummy" not in tool_call]
                if len(actionable_tool_calls) > 1:
                    print("   → Warning: model returned multiple actions in one step; executing only the first.")
                    for extra_tool_call in actionable_tool_calls[1:]:
                        messages.append(
                            ToolMessage(
                                tool_call_id=extra_tool_call["id"],
                                name=extra_tool_call["name"],
                                content=json.dumps(
                                    {
                                        "status": "error",
                                        "message": (
                                            "Only one `call_tool_with_reason` action is allowed per assistant turn. "
                                            "This extra action was ignored. Wait for the next step before issuing another action."
                                        ),
                                    }
                                ),
                            )
                        )
                    tool_calls = [actionable_tool_calls[0]]

                for tool_call in tool_calls:
                    if "dummy" in tool_call:
                        continue

                    reason = str(tool_call["args"].get("reason", "")).strip()
                    action_name = str(tool_call["args"].get("tool_name", "")).strip()
                    action_args = tool_call["args"].get("tool_args", {})

                    if not reason or not action_name or not isinstance(action_args, dict):
                        result_str = json.dumps(
                            {
                                "status": "error",
                                "message": (
                                    "You must call `call_tool_with_reason` with non-empty `reason`, "
                                    "valid `tool_name`, and object-valued `tool_args`."
                                ),
                            }
                        )
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=result_str,
                            )
                        )
                        continue

                    print(f"   → Reason: {reason}")

                    if action_name == "finish_task":
                        result_str = await finish_task(mcp_client, action_args.get("summary", "Task finished."))
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=format_action_result(action_name, reason, result_str),
                            )
                        )
                        print(f"\nFinal Report: {action_args.get('summary', 'Task finished.')}")
                        return

                    result_str = await execute_mcp_tool(
                        mcp_client,
                        action_name,
                        action_args,
                        local_tool_handlers={
                            "check_stacking_status": lambda client, tool_args: check_stacking_status(client),
                        },
                        log_prefix="Agent Executing",
                    )
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                            content=format_action_result(action_name, reason, result_str),
                        )
                    )

                iteration += 1
            else:
                print(f"Safety Break: Reached the maximum of {args.max_steps} iterations.")

    except Exception as exc:
        print(f"Execution failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
