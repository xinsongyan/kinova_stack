from __future__ import annotations

import json
import re

from langchain_core.messages import ToolMessage


async def handle_prompt_tool_call(mcp_client, func_name, args):
    """Route `get_prompt_*` tool calls to MCP `get_prompt`."""
    prompt_name = func_name.replace("get_prompt_", "", 1)
    try:
        result = await mcp_client.get_prompt(prompt_name, args)
        if result and hasattr(result, "messages") and result.messages:
            content = result.messages[0].content
            text = content.text if hasattr(content, "text") else str(content)
            print(f"   → Read SOP Prompt: {prompt_name}")
            return text
        return json.dumps({"status": "error", "message": "Prompt returned empty messages."})
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Failed to get prompt: {exc}"})


async def execute_mcp_tool(mcp_client, tool_name: str, args: dict, local_tool_handlers=None, log_prefix="Agent Executing") -> str:
    """Execute a local helper, prompt tool, or MCP tool and stringify the result."""
    print(f"\n{log_prefix}: {tool_name}({args})")

    if local_tool_handlers and tool_name in local_tool_handlers:
        return await local_tool_handlers[tool_name](mcp_client, args)

    if tool_name.startswith("get_prompt_"):
        return await handle_prompt_tool_call(mcp_client, tool_name, args)

    try:
        result = await mcp_client.call_tool(tool_name, args)
        result_json = json.dumps(result.structured_content or {}, default=str)
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "... [truncated]"
        print(f"   → Success: {result_json[:200]}...")
        return result_json
    except Exception as exc:
        error_msg = json.dumps({"status": "error", "message": str(exc)})
        print(f"   → Failed: {error_msg}")
        return error_msg


async def finish_task(_mcp_client, summary: str) -> str:
    """Return a structured completion marker for the client loop."""
    result = json.dumps({"status": "finished", "summary": summary})
    print(f"   → finish_task: {summary}")
    return result


def _repair_tool_args_json(raw_args: str) -> str:
    """Best-effort cleanup for malformed JSON-ish tool arguments from the model."""
    repaired = raw_args.replace("｜", "|")
    repaired = re.sub(r"[\u4e00-\u9fff]", "", repaired)
    repaired = re.sub(r"(?<=[:\[,])\s*[A-Za-z_\u0080-\uFFFF][A-Za-z0-9_\u0080-\uFFFF]*\s*(?=[,\]\}])", " 0 ", repaired)
    repaired = re.sub(r",\s*([\]\}])", r"\1", repaired)
    return repaired.strip()


def _strip_json_code_fence(content: str) -> str:
    """Remove a surrounding Markdown code fence when the model wraps JSON in one."""
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _normalize_json_tool_call(entry: dict, iteration: int, index: int) -> dict:
    """Convert a JSON-emitted tool call object into the internal tool-call format."""
    function_block = entry.get("function")
    if isinstance(function_block, dict):
        tool_name = function_block.get("name")
        raw_args = function_block.get("arguments", function_block.get("parameters", {}))
    else:
        tool_name = entry.get("name", entry.get("tool_name"))
        raw_args = entry.get("args", entry.get("arguments", entry.get("parameters", {})))

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("missing tool name")

    if isinstance(raw_args, str):
        cleaned_args = _repair_tool_args_json(raw_args)
        raw_args = json.loads(cleaned_args) if cleaned_args else {}

    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise ValueError("tool arguments must be a JSON object")

    return {
        "name": tool_name.strip(),
        "args": raw_args,
        "id": str(entry.get("id") or f"fallback_{iteration}_{index}_{tool_name.strip()}"),
    }


def _parse_json_tool_calls(content: str, iteration: int) -> tuple[list[dict], list[ToolMessage]]:
    """Parse raw JSON tool calls from providers that emit them in assistant content."""
    tool_calls = []
    tool_messages = []
    candidate = _strip_json_code_fence(content)

    if not candidate or candidate[0] not in "[{":
        return tool_calls, tool_messages

    try:
        payload = json.loads(candidate)
    except Exception:
        return tool_calls, tool_messages

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return tool_calls, tool_messages

    print("   [Fallback Parser] Found JSON tool call content. Extracting...")

    for index, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            err_msg = (
                "TOOL PARSE ERROR: JSON tool call payload must be an object or a list of objects. "
                "Resend the next action as a valid structured tool call."
            )
            tool_messages.append(
                ToolMessage(
                    tool_call_id=f"err_{iteration}_json_tool_parse_{index}",
                    name="tool_parse_error",
                    content=json.dumps({"status": "error", "message": err_msg}),
                )
            )
            continue

        try:
            tool_calls.append(_normalize_json_tool_call(entry, iteration, index))
        except Exception as exc:
            err_msg = (
                f"JSON PARSE ERROR: Could not interpret tool call object #{index}: {exc}. "
                "Include `name` plus `parameters`/`arguments` as a JSON object."
            )
            print(f"   [Fallback Parser] {err_msg}")
            tool_messages.append(
                ToolMessage(
                    tool_call_id=f"err_{iteration}_json_tool_parse_{index}",
                    name="tool_parse_error",
                    content=json.dumps({"status": "error", "message": err_msg}),
                )
            )

    return tool_calls, tool_messages


def parse_raw_tool_calls(content: str, iteration: int) -> tuple[list[dict], list[ToolMessage]]:
    """Fallback parser for models that emit raw tool tags or JSON instead of structured calls."""
    tool_calls = []
    tool_messages = []

    if not content:
        return tool_calls, tool_messages

    json_tool_calls, json_tool_messages = _parse_json_tool_calls(content, iteration)
    if json_tool_calls or json_tool_messages:
        return json_tool_calls, json_tool_messages

    normalized_content = content.replace("｜", "|")
    if "<|tool▁call▁begin|>" not in normalized_content:
        return tool_calls, tool_messages

    print("   [Fallback Parser] Found raw tool call tags. Extracting...")
    sanitized_content = re.sub(r"[\u4e00-\u9fff]", "", normalized_content)
    raw_calls = re.findall(
        r"<\|tool▁call▁begin\|>(.*?)<\|tool▁sep\|>(.*?)<\|tool▁call▁end\|>",
        sanitized_content,
        re.DOTALL,
    )

    if not raw_calls:
        err_msg = (
            "TOOL PARSE ERROR: Raw tool-call tags were detected, but no valid tool calls could be extracted. "
            "Resend the next action as a valid structured tool call with valid JSON arguments only."
        )
        print(f"   [Fallback Parser] {err_msg}")
        tool_messages.append(
            ToolMessage(
                tool_call_id=f"err_{iteration}_raw_tool_parse",
                name="tool_parse_error",
                content=json.dumps({"status": "error", "message": err_msg}),
            )
        )
        return tool_calls, tool_messages

    for tool_name, tool_args_raw in raw_calls:
        try:
            cleaned_args = _repair_tool_args_json(tool_args_raw)
            tool_calls.append(
                {
                    "name": tool_name.strip(),
                    "args": json.loads(cleaned_args),
                    "id": f"fallback_{iteration}_{tool_name.strip()}",
                }
            )
        except Exception as exc:
            err_msg = (
                f"JSON PARSE ERROR: Your tool call for '{tool_name.strip()}' was malformed: {exc}. "
                "Please resend only valid JSON."
            )
            print(f"   [Fallback Parser] {err_msg}")
            tool_messages.append(
                ToolMessage(
                    tool_call_id=f"err_{iteration}_{tool_name.strip()}",
                    name=tool_name.strip(),
                    content=json.dumps({"status": "error", "message": err_msg}),
                )
            )

    return tool_calls, tool_messages


def build_retry_tool_message(iteration: int, reason: str) -> ToolMessage:
    """Create a tool-like error message that nudges the model to retry with valid tools."""
    return ToolMessage(
        tool_call_id=f"err_{iteration}_retry",
        name="agent_retry_required",
        content=json.dumps(
            {
                "status": "error",
                "message": (
                    f"{reason} Use a valid tool call next, or call finish_task(summary=...) if the task is complete."
                ),
            }
        ),
    )
