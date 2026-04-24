from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from kinova_middleware.llm_clients.rate_limits import invoke_with_rate_limit_retry
from kinova_middleware.llm_clients.tool_dispatch import (
    build_retry_tool_message,
    execute_mcp_tool,
    finish_task,
    parse_raw_tool_calls,
)
from kinova_middleware.llm_clients.tool_schema import (
    bind_model_tools,
    build_action_reference,
    build_reasoned_action_tool,
    format_action_result,
    load_tools_and_prompts_from_mcp,
)


LocalToolHandler = Callable[[Any, dict], Awaitable[str]]


@dataclass(slots=True)
class ReasonedAgentSession:
    tools_schema: list[dict]
    action_reference: str
    llm_with_tools: Any
    messages: list


@dataclass(slots=True)
class WorkflowRunResult:
    stop_reason: str
    model_summary: str
    messages: list


async def build_reasoned_agent_session(
    mcp_client,
    llm,
    *,
    system_prompt: str,
    task: str,
    extra_tools: list[dict] | None = None,
    tool_choice: str = "required",
    skip_reset_scene: bool = True,
) -> ReasonedAgentSession:
    tools_schema = await load_tools_and_prompts_from_mcp(
        mcp_client,
        extra_tools=extra_tools,
        skip_reset_scene=skip_reset_scene,
    )
    action_reference = build_action_reference(tools_schema)
    llm_with_tools = bind_model_tools(
        llm,
        [build_reasoned_action_tool(tools_schema)],
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=action_reference),
        HumanMessage(content=task),
    ]
    return ReasonedAgentSession(
        tools_schema=tools_schema,
        action_reference=action_reference,
        llm_with_tools=llm_with_tools,
        messages=messages,
    )


async def run_reasoned_agent_loop(
    mcp_client,
    llm_with_tools,
    messages: list,
    *,
    max_steps: int,
    local_tool_handlers: dict[str, LocalToolHandler] | None = None,
    finish_summary_default: str = "Task stopped before finish_task was called.",
    log_prefix: str = "Agent Executing",
) -> WorkflowRunResult:
    handlers = local_tool_handlers or {}

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
                result_str = await finish_task(mcp_client, model_summary)
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                        content=format_action_result(action_name, reason, result_str),
                    )
                )
                return WorkflowRunResult(
                    stop_reason="finish_task",
                    model_summary=model_summary,
                    messages=messages,
                )

            result_str = await execute_mcp_tool(
                mcp_client,
                action_name,
                action_args,
                local_tool_handlers=handlers,
                log_prefix=log_prefix,
            )
            messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    content=format_action_result(action_name, reason, result_str),
                )
            )

    return WorkflowRunResult(
        stop_reason="max_steps",
        model_summary=finish_summary_default,
        messages=messages,
    )
