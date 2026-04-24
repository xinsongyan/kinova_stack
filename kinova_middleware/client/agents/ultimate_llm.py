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

if __package__ in (None, ""):
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from fastmcp import Client
from langchain_openai import ChatOpenAI
from kinova_middleware.client.local_tools.status_reporting import (
    check_sorting_progress,
    check_stacking_status,
    verify_object_lift,
)
from kinova_middleware.client.local_tools.tool_defs import (
    CHECK_SORTING_STATUS_TOOL,
    CHECK_STACKING_STATUS_TOOL,
    FINISH_TASK_TOOL,
    VERIFY_OBJECT_LIFT_TOOL,
)
from kinova_middleware.client.runtime.workflow_runner import (
    build_reasoned_agent_session,
    run_reasoned_agent_loop,
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
- Use only tools present in the action reference.
- Never invent `pick_cube`, `place_cube`, or any other missing helper tools.
- Use the single structured wrapper tool `call_tool_with_reason` for every action.
- For every action, provide:
  - `reason`: one short sentence explaining what you are about to do and why
  - `tool_name`: the exact action name from the action reference
  - `tool_args`: a JSON object containing the arguments for that action
- Example:
  - `reason`: "I am going to check the current scene state so I know which workflow-specific action should come next."
  - `tool_name`: `check_sorting_status`
  - `tool_args`: `{}`
- Exactly one `call_tool_with_reason` call is allowed per assistant turn.
- Do not bundle multiple actions into one response. Wait for the tool result before deciding the next action.
- Do not call more than one task prompt unless the user changes the task.
- Keep execution sequential and grounded in tool outputs.
"""


def build_local_tools() -> list[dict]:
    """Return local tool definitions exposed only inside this client."""
    return [CHECK_SORTING_STATUS_TOOL, CHECK_STACKING_STATUS_TOOL, VERIFY_OBJECT_LIFT_TOOL, FINISH_TASK_TOOL]


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
            session = await build_reasoned_agent_session(
                mcp_client,
                llm,
                system_prompt=SYSTEM_PROMPT,
                task=task,
                extra_tools=build_local_tools(),
                tool_choice="required",
                skip_reset_scene=True,
            )
            print(
                "Successfully loaded "
                f"{len(session.tools_schema)} callable tools (including prompt wrappers and local helpers)."
            )

            run_result = await run_reasoned_agent_loop(
                mcp_client,
                session.llm_with_tools,
                session.messages,
                max_steps=MAX_AGENT_STEPS,
                local_tool_handlers={
                    "check_sorting_status": lambda client, tool_args: check_sorting_progress(client),
                    "check_stacking_status": lambda client, tool_args: check_stacking_status(client),
                    "verify_object_lift": lambda client, tool_args: verify_object_lift(
                        client,
                        tool_args.get("body_name", ""),
                        float(tool_args.get("min_height", 0.12)),
                    ),
                },
                finish_summary_default="Task stopped before finish_task was called.",
            )

            if run_result.stop_reason == "finish_task":
                print(f"\nFinal Report: {run_result.model_summary}")
                return

            if run_result.stop_reason == "max_steps":
                print(f"Safety Break: Reached the maximum of {MAX_AGENT_STEPS} iterations.")

    except Exception as exc:
        print(f"Execution failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
