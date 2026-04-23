#!/usr/bin/env python3
"""
Grab-shapes LLM client with official verification reporting.

This client uses the same routing prompt as `ultimate_llm.py`, but it is
preconfigured for the `grab_shapes` workflow and produces a final report from
actual tool outputs and live scene state instead of trusting the model's own
summary text.

Examples:
  python kinova_middleware/llm_clients/grab_shapes.py
  python kinova_middleware/llm_clients/grab_shapes.py --model openai/gpt-oss-120b
  python kinova_middleware/llm_clients/grab_shapes.py --runs 5
"""

import argparse
import asyncio
import json
import os
import sys
import time

from fastmcp import Client
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from helper_functions import (
    FINISH_TASK_TOOL,
    VERIFY_OBJECT_LIFT_TOOL,
    bind_model_tools,
    build_action_reference,
    build_reasoned_action_tool,
    build_retry_tool_message,
    collect_grab_shapes_report,
    execute_mcp_tool,
    finish_task,
    format_action_result,
    format_grab_shapes_report,
    invoke_with_rate_limit_retry,
    load_tools_and_prompts_from_mcp,
    parse_raw_tool_calls,
    record_grab_verification,
    reset_scene_if_available,
    verify_object_lift,
)


DEFAULT_MODEL_NAME = "moonshotai/kimi-k2-instruct-0905"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_SCENE_NAME = "shapes.xml"
DEFAULT_TASK = "Grab and lift the cylinder, box, sphere. start by picking up the cylinder. then pick up and drop the box then pick up the sphere."
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
   - `verify_object_lift` for grab/lift verification
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
  - `reason`: "I am going to get the position of the cylinder so I can plan the grasp."
  - `tool_name`: `get_object_pose`
  - `tool_args`: `{"body_name": "cylinder"}`
- Exactly one `call_tool_with_reason` call is allowed per assistant turn.
- Do not bundle multiple actions into one response. Wait for the tool result before deciding the next action.
- Do not call more than one task prompt unless the user changes the task.
- Keep execution sequential and grounded in tool outputs.

Completion rules for `grab_shapes`:
- After each lift attempt, call `verify_object_lift(body_name='...')` before claiming success.
- Do not claim an object was picked up unless `verify_object_lift` returned PASS.
- Call `finish_task(summary=...)` only after you have finished the task or cannot continue safely.
"""


def build_local_tools() -> list[dict]:
    """Return client-only helper tools for the grab_shapes workflow."""
    return [VERIFY_OBJECT_LIFT_TOOL, FINISH_TASK_TOOL]


async def run_single_benchmark(
    mcp_client,
    llm: ChatOpenAI,
    task: str,
    max_steps: int,
    min_lift_height: float,
    tool_choice: str,
) -> dict:
    """Run one grab_shapes episode and return model and official results."""
    verification_history: dict = {}
    tools_schema = await load_tools_and_prompts_from_mcp(
        mcp_client,
        extra_tools=build_local_tools(),
        skip_reset_scene=True,
    )
    action_reference = build_action_reference(tools_schema)
    llm_with_tools = bind_model_tools(
        llm,
        [build_reasoned_action_tool(tools_schema)],
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=action_reference),
        HumanMessage(content=task),
    ]
    model_summary = "Task stopped before finish_task was called."
    stop_reason = "max_steps"

    async def tracked_verify(client, tool_args):
        body_name = tool_args.get("body_name", "")
        threshold = float(tool_args.get("min_height", min_lift_height))
        result = await verify_object_lift(client, body_name, threshold)
        record_grab_verification(verification_history, body_name, result, threshold)
        return result

    for iteration in range(1, max_steps + 1):
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
                model_summary = action_args.get("summary", "Task finished.")
                stop_reason = "finish_task"
                result_str = await finish_task(mcp_client, model_summary)
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                        content=format_action_result(action_name, reason, result_str),
                    )
                )
                official_data = await collect_grab_shapes_report(
                    mcp_client,
                    verification_history,
                    default_min_height=min_lift_height,
                )
                return {
                    "stop_reason": stop_reason,
                    "model_summary": model_summary,
                    "official_data": official_data,
                    "official_report": format_grab_shapes_report(official_data),
                }

            result_str = await execute_mcp_tool(
                mcp_client,
                action_name,
                action_args,
                local_tool_handlers={
                    "verify_object_lift": tracked_verify,
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

    official_data = await collect_grab_shapes_report(
        mcp_client,
        verification_history,
        default_min_height=min_lift_height,
    )
    return {
        "stop_reason": stop_reason,
        "model_summary": model_summary,
        "official_data": official_data,
        "official_report": format_grab_shapes_report(official_data),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Dedicated grab_shapes LLM client with official reporting.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task instruction for the model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name to benchmark.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="FastMCP server URL.")
    parser.add_argument("--runs", type=int, default=1, help="Number of fresh runs to execute.")
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS, help="Maximum agent iterations per run.")
    parser.add_argument(
        "--scene-number",
        type=int,
        default=None,
        help=f"Optional 1-based scene number to load before each run. Defaults to {DEFAULT_SCENE_NAME}.",
    )
    parser.add_argument(
        "--tool-choice",
        default="required",
        choices=["required", "auto", "none", "default"],
        help="Tool-choice mode passed to the OpenAI-compatible model API. Use 'required' for vLLM-like servers that reject implicit/auto mode.",
    )
    parser.add_argument(
        "--min-lift-height",
        type=float,
        default=0.12,
        help="Minimum Z height required for an official lift PASS.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        sys.exit("Error: --runs must be at least 1.")
    if args.max_steps < 1:
        sys.exit("Error: --max-steps must be at least 1.")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("Error: NVIDIA_API_KEY not set in environment.")

    print(f"Initializing grab_shapes agent ({args.model}) via NVIDIA...")
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        api_key=api_key,
        base_url=args.base_url,
    )

    benchmark_rows = []

    print(f"Connecting to FastMCP Server at {args.server_url} ...")
    try:
        async with Client(args.server_url) as mcp_client:
            available_tool_names = {tool.name for tool in await mcp_client.list_tools()}
            print(f"Connected! Server exposes {len(available_tool_names)} tools.")

            for run_number in range(1, args.runs + 1):
                print("")
                print("=" * 72)
                print(f"Run {run_number}/{args.runs} | model={args.model} | tool_choice={args.tool_choice}")
                print("=" * 72)
                await reset_scene_if_available(
                    mcp_client,
                    available_tool_names,
                    scene_name=DEFAULT_SCENE_NAME,
                    scene_number=args.scene_number,
                    run_number=run_number,
                    total_runs=args.runs,
                )

                started_at = time.perf_counter()
                result = await run_single_benchmark(
                    mcp_client,
                    llm,
                    args.task,
                    args.max_steps,
                    args.min_lift_height,
                    args.tool_choice,
                )
                duration_s = time.perf_counter() - started_at
                verified_count = result["official_data"]["verified_count"]
                target_count = result["official_data"]["target_count"]

                print("")
                print(f"Model Summary: {result['model_summary']}")
                print(result["official_report"])
                print(f"Run Duration: {duration_s:.2f}s")

                benchmark_rows.append(
                    {
                        "run_number": run_number,
                        "stop_reason": result["stop_reason"],
                        "verified_count": verified_count,
                        "target_count": target_count,
                        "duration_s": duration_s,
                    }
                )

            if len(benchmark_rows) > 1:
                print("")
                print("BENCHMARK SUMMARY:")
                for row in benchmark_rows:
                    print(
                        f"- Run {row['run_number']}: officially picked up "
                        f"{row['verified_count']}/{row['target_count']} objects, "
                        f"stop_reason={row['stop_reason']}, duration={row['duration_s']:.2f}s"
                    )

    except Exception as exc:
        print(f"Execution failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
