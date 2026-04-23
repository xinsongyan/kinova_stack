import json
import os
import random
import re
import time

from langchain_core.messages import ToolMessage


CHECK_SORTING_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_sorting_status",
        "description": "Analyze the scene to see which cubes are sorted, which are at start, and which are out of workspace.",
        "parameters": {"type": "object", "properties": {}},
    },
}

CHECK_STACKING_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_stacking_status",
        "description": "Analyze the cubes and report which cube is stacked on top of which other cube based on z height and xy alignment.",
        "parameters": {"type": "object", "properties": {}},
    },
}

GRAB_SHAPES_TARGETS = ("box", "sphere", "cylinder")

VERIFY_OBJECT_LIFT_TOOL = {
    "type": "function",
    "function": {
        "name": "verify_object_lift",
        "description": "Check whether a named object has been lifted high enough after a grasp.",
        "parameters": {
            "type": "object",
            "properties": {
                "body_name": {
                    "type": "string",
                    "description": "The object name to verify, for example box, sphere, or cylinder.",
                },
                "min_height": {
                    "type": "number",
                    "description": "Minimum Z height that counts as a successful lift. Default is 0.12.",
                },
            },
            "required": ["body_name"],
        },
    },
}

FINISH_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": "Call this when the task is truly complete or cannot continue safely. This is the only normal way to end the agent loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short final summary of what was completed or why the task stopped.",
                }
            },
            "required": ["summary"],
        },
    },
}

DEFAULT_LLM_RATE_LIMIT_RETRIES = 8
DEFAULT_LLM_RATE_LIMIT_BASE_DELAY_S = 2.0
DEFAULT_LLM_RATE_LIMIT_MAX_DELAY_S = 60.0
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCENES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "scenes"))


def discover_local_scene_names() -> list[str]:
    """Return the locally available scene basenames in server selection order."""
    if not os.path.isdir(_SCENES_DIR):
        return []
    return [
        name
        for name in sorted(os.listdir(_SCENES_DIR))
        if name.endswith((".xml", ".mjcf"))
    ]


def resolve_local_scene_number(scene_name: str) -> int:
    """Resolve a scene basename to the 1-based number used by reset_scene()."""
    normalized_name = os.path.basename(scene_name)
    scene_names = discover_local_scene_names()
    if normalized_name not in scene_names:
        raise ValueError(
            f"Scene '{normalized_name}' was not found in {_SCENES_DIR}. "
            f"Available scenes: {', '.join(scene_names) if scene_names else 'none'}."
        )
    return scene_names.index(normalized_name) + 1


async def reset_scene_if_available(
    mcp_client,
    available_tool_names: set[str],
    *,
    scene_name: str | None = None,
    scene_number: int | None = None,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> None:
    """Reset or hot-swap the scene before an agent run when supported."""
    if "reset_scene" not in available_tool_names:
        print("Warning: reset_scene() is not available, so the scene will not auto-reset.")
        return

    selected_scene_number = scene_number
    if selected_scene_number is None and scene_name is not None:
        selected_scene_number = resolve_local_scene_number(scene_name)

    tool_args = {}
    if selected_scene_number is not None:
        tool_args["scene_number"] = selected_scene_number

    result = await mcp_client.call_tool("reset_scene", tool_args)
    data = result.structured_content or {}
    status = data.get("status", "unknown")
    message = data.get("message", "No reset_scene message returned.")

    prefix = f"[Run {run_number}] " if run_number is not None else ""
    print(f"{prefix}reset_scene -> {status}: {message}")


async def load_tools_and_prompts_from_mcp(mcp_client, extra_tools=None, skip_reset_scene=True):
    """
    Load MCP tools and prompts, and optionally append local client-only tools.

    By default, `reset_scene` is hidden from the LLM tool schema so the model
    cannot reset the environment on its own. Clients can still call the MCP tool
    directly outside the schema when they need a deterministic benchmark reset.
    """
    print("Fetching tools from MCP server...")
    try:
        mcp_tools = await mcp_client.list_tools()
        openai_tools = []
        for tool in mcp_tools:
            if skip_reset_scene and tool.name == "reset_scene":
                continue
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {},
                    },
                }
            )
            print(f" - Loaded tool: {tool.name}")

        print("Fetching prompts from MCP server...")
        mcp_prompts = await mcp_client.list_prompts()
        for prompt in mcp_prompts:
            print(f" - Loaded prompt template as tool: get_prompt_{prompt.name}")
            properties = {}
            required = []
            if prompt.arguments:
                for arg in prompt.arguments:
                    properties[arg.name] = {
                        "type": "string",
                        "description": arg.description or f"The {arg.name} to insert into the prompt",
                    }
                    if arg.required:
                        required.append(arg.name)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"get_prompt_{prompt.name}",
                        "description": prompt.description or f"Get the Standard Operating Procedure (SOP) for {prompt.name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )

        if extra_tools:
            openai_tools.extend(extra_tools)

        return openai_tools
    except Exception as e:
        print(f"Failed to load tools and prompts: {e}")
        return []


def bind_model_tools(
    llm,
    tools_schema: list[dict],
    tool_choice: str | None = "required",
    parallel_tool_calls: bool | None = None,
):
    """Bind tools to an LLM with a provider-compatible tool choice mode.

    The default `required` mode avoids OpenAI-compatible servers such as vLLM
    that reject implicit/auto tool choice unless extra server flags are set.
    Use `tool_choice="default"` to let the provider choose its own default.
    """
    normalized_tool_choice = tool_choice
    if isinstance(normalized_tool_choice, str):
        normalized_tool_choice = normalized_tool_choice.strip().lower()
        if normalized_tool_choice in {"", "default"}:
            normalized_tool_choice = None

    bind_kwargs = {}
    if normalized_tool_choice is not None:
        bind_kwargs["tool_choice"] = normalized_tool_choice
    if parallel_tool_calls is not None:
        bind_kwargs["parallel_tool_calls"] = parallel_tool_calls

    return llm.bind_tools([tool["function"] for tool in tools_schema], **bind_kwargs)


def build_reasoned_action_tool(tools_schema: list[dict]) -> dict:
    """Return a single wrapper tool that requires a reason for every action."""
    action_names = [tool["function"]["name"] for tool in tools_schema]
    return {
        "type": "function",
        "function": {
            "name": "call_tool_with_reason",
            "description": (
                "Call exactly one action from the action reference. "
                "You must provide a short reason, the exact tool name, and a JSON object of tool arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining what you are about to do and why.",
                    },
                    "tool_name": {
                        "type": "string",
                        "enum": action_names,
                        "description": "Exact action name from the action reference.",
                    },
                    "tool_args": {
                        "type": "object",
                        "description": "JSON object containing the arguments for tool_name. Use {} when there are no arguments.",
                        "additionalProperties": True,
                    },
                },
                "required": ["reason", "tool_name", "tool_args"],
            },
        },
    }


def build_action_reference(tools_schema: list[dict]) -> str:
    """Render the available actions and schemas as prompt text for the wrapper tool."""
    lines = [
        "Action reference:",
        "Use only the action names and argument schemas listed below.",
        "For each action, call `call_tool_with_reason` with `reason`, `tool_name`, and `tool_args`.",
    ]
    for tool in tools_schema:
        fn = tool["function"]
        params = json.dumps(fn.get("parameters", {}), ensure_ascii=True, separators=(",", ":"))
        lines.append(f"- {fn['name']}: {fn.get('description', '')}")
        lines.append(f"  parameters={params}")
    return "\n".join(lines)


def format_action_result(tool_name: str, reason: str, result_str: str) -> str:
    """Return the wrapper-tool response content sent back to the model."""
    return (
        f"Reason: {reason}\n"
        f"Executed action: {tool_name}\n"
        f"Result:\n{result_str}"
    )


def _coerce_retry_after_seconds(value) -> float | None:
    """Convert a Retry-After-like header value to seconds when possible."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds > 0 else None

    if isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            return None
        return seconds if seconds > 0 else None

    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of retry delay from provider exceptions."""
    for attr_name in ("retry_after", "retry_after_seconds"):
        retry_after = _coerce_retry_after_seconds(getattr(exc, attr_name, None))
        if retry_after is not None:
            return retry_after

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        for header_name in (
            "retry-after",
            "Retry-After",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ):
            retry_after = _coerce_retry_after_seconds(headers.get(header_name))
            if retry_after is not None:
                return retry_after

    return None


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True when the exception looks like a provider-side rate limit."""
    if getattr(exc, "status_code", None) == 429:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("status") == 429:
        return True

    message = str(exc).lower()
    return (
        "too many requests" in message
        or "rate limit" in message
        or "ratelimit" in message
        or "error code: 429" in message
        or "{'status': 429" in message
    )


def invoke_with_rate_limit_retry(
    llm_with_tools,
    messages,
    *,
    max_retries: int = DEFAULT_LLM_RATE_LIMIT_RETRIES,
    base_delay_s: float = DEFAULT_LLM_RATE_LIMIT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_LLM_RATE_LIMIT_MAX_DELAY_S,
):
    """Invoke the model, retrying the same step on provider 429 responses."""
    retry_count = 0

    while True:
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if retry_count >= max_retries:
                raise RuntimeError(
                    f"LLM rate-limit retries exhausted after {max_retries} retries: {exc}"
                ) from exc

            retry_count += 1
            retry_after_s = _extract_retry_after_seconds(exc)
            if retry_after_s is None:
                delay_s = min(max_delay_s, base_delay_s * (2 ** (retry_count - 1)))
            else:
                delay_s = min(max_delay_s, max(base_delay_s, retry_after_s))

            jitter_s = min(1.0, delay_s * 0.1) * random.random()
            wait_s = delay_s + jitter_s
            print(
                "   → Model API rate limited (429). "
                f"Waiting {wait_s:.1f}s before retrying the same step "
                f"[retry {retry_count}/{max_retries}]."
            )
            time.sleep(wait_s)


async def handle_prompt_tool_call(mcp_client, func_name, args):
    """
    Route `get_prompt_*` tool calls to MCP `get_prompt`.
    """
    prompt_name = func_name.replace("get_prompt_", "", 1)
    try:
        result = await mcp_client.get_prompt(prompt_name, args)
        if result and hasattr(result, "messages") and result.messages:
            content = result.messages[0].content
            text = content.text if hasattr(content, "text") else str(content)
            print(f"   → Read SOP Prompt: {prompt_name}")
            return text
        return json.dumps({"status": "error", "message": "Prompt returned empty messages."})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to get prompt: {e}"})


async def check_sorting_progress(mcp_client) -> str:
    """Analyze the current scene and report sorting progress to the LLM."""
    bin_centers = {
        "red_bin_target": (0.0, 0.35),
        "blue_bin_target": (0.0, -0.35),
    }
    bin_tolerance = 0.10
    workspace = {
        "red": {"x": (0.18, 0.30), "y": (-0.25, 0.25)},
        "blue": {"x": (-0.30, -0.18), "y": (-0.25, 0.25)},
    }

    try:
        res = await mcp_client.call_tool("get_object_pose", {"body_name": "all"})
        data = res.structured_content or {}
        objects = data.get("objects", [])

        report = []
        actionable = []

        for obj in objects:
            name = obj["body_name"]
            if "cube" not in name.lower():
                continue

            pos = obj["position"]
            x, y = pos["x"], pos["y"]
            color = "red" if "red" in name.lower() else "blue"
            bin_name = f"{color}_bin_target"
            bx, by = bin_centers[bin_name]

            if abs(x - bx) < bin_tolerance and abs(y - by) < bin_tolerance:
                report.append(f"- {name}: Correctly Sorted (In {bin_name})")
                continue

            bounds = workspace[color]
            if bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1]:
                report.append(f"- {name}: At Starting Position (In {color}_zone - Ready to pick)")
                actionable.append(name)
                continue

            report.append(f"- {name}: OUT OF WORKSPACE / MISPLACED (x={x:.3f}, y={y:.3f})")

        summary = "CURRENT SORTING STATUS REPORT:\n" + ("\n".join(report) if report else "No cubes detected.")
        if not actionable and any("Sorted" not in line for line in report):
            summary += "\n\nWARNING: No actionable cubes found in their workspaces. Remaining cubes are misplaced."
        elif not actionable:
            summary += "\n\nMISSION STATUS: ALL CUBES SORTED SUCCESSFULLY."

        summary += "\n\nCRITICAL SAFETY RULE: Never attempt to pick a cube marked 'OUT OF WORKSPACE'. Focus only on cubes in their designated start zones."
        print(f"\n[check_sorting_status summary]:\n{summary}\n")
        return summary
    except Exception as e:
        return f"Error checking progress: {str(e)}"


async def check_stacking_status(mcp_client) -> str:
    """Analyze cube positions and report whether any cube is stacked on another cube."""
    xy_tolerance = 0.05
    min_vertical_gap = 0.045

    try:
        res = await mcp_client.call_tool("get_object_pose", {"body_name": "all"})
        data = res.structured_content or {}
        objects = data.get("objects", [])
        cubes = [obj for obj in objects if "cube" in obj.get("body_name", "").lower()]

        if not cubes:
            return "CURRENT STACKING STATUS REPORT:\nNo cubes detected."

        report = ["CURRENT STACKING STATUS REPORT:"]
        for cube in sorted(cubes, key=lambda item: item["position"]["z"]):
            pos = cube["position"]
            report.append(f"- {cube['body_name']}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")

        relations = []
        for top_cube in cubes:
            top_pos = top_cube["position"]
            top_hh = top_cube.get("size", [0.03, 0.03, 0.03])[2] if len(top_cube.get("size", [])) >= 3 else 0.03
            best_candidate = None

            for bottom_cube in cubes:
                if bottom_cube["body_name"] == top_cube["body_name"]:
                    continue

                bottom_pos = bottom_cube["position"]
                bottom_hh = bottom_cube.get("size", [0.03, 0.03, 0.03])[2] if len(bottom_cube.get("size", [])) >= 3 else 0.03
                dx = abs(top_pos["x"] - bottom_pos["x"])
                dy = abs(top_pos["y"] - bottom_pos["y"])
                dz = top_pos["z"] - bottom_pos["z"]
                expected_gap = top_hh + bottom_hh

                if dz <= min_vertical_gap or dx > xy_tolerance or dy > xy_tolerance:
                    continue

                score = abs(dz - expected_gap) + dx + dy
                if best_candidate is None or score < best_candidate["score"]:
                    best_candidate = {
                        "top": top_cube["body_name"],
                        "bottom": bottom_cube["body_name"],
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "score": score,
                    }

            if best_candidate is not None:
                relations.append(best_candidate)

        unique_relations = {}
        for relation in relations:
            key = (relation["top"], relation["bottom"])
            if key not in unique_relations or relation["score"] < unique_relations[key]["score"]:
                unique_relations[key] = relation

        if unique_relations:
            report.append("")
            report.append("DETECTED STACK RELATIONS:")
            for relation in sorted(unique_relations.values(), key=lambda item: item["dz"], reverse=True):
                report.append(
                    f"- {relation['top']} is on top of {relation['bottom']} "
                    f"(dx={relation['dx']:.3f}, dy={relation['dy']:.3f}, dz={relation['dz']:.3f})"
                )
            if ("blue_cube", "red_cube") in unique_relations:
                report.append("")
                report.append("MISSION STATUS: blue_cube is stacked on red_cube.")
        else:
            report.append("")
            report.append("MISSION STATUS: No cube-on-cube stack detected yet.")

        summary = "\n".join(report)
        print(f"\n[check_stacking_status summary]:\n{summary}\n")
        return summary
    except Exception as e:
        return f"Error checking stacking status: {str(e)}"


async def verify_object_lift(mcp_client, body_name: str, min_height: float = 0.12) -> str:
    """Check whether a named object has been lifted high enough above the table."""
    canonical_body_name = str(body_name or "").strip()
    try:
        result = await mcp_client.call_tool("get_object_pose", {"body_name": canonical_body_name})
        data = result.structured_content or {}
        if data.get("status") == "error":
            return f"LIFT CHECK ERROR: {data.get('message', 'Unknown error')}"

        position = data.get("position", {})
        z = float(position.get("z", 0.0))
        lifted = z >= min_height

        summary = (
            f"LIFT STATUS for '{canonical_body_name}': z={z:.3f} m, threshold={min_height:.3f} m. "
            f"{'PASS: object is lifted high enough.' if lifted else 'FAIL: object is still too low.'}"
        )
        print(f"\n[verify_object_lift summary]:\n{summary}\n")
        return summary
    except Exception as e:
        return f"LIFT CHECK ERROR for '{canonical_body_name}': {str(e)}"


def record_grab_verification(verification_history: dict, body_name: str, result: str, min_height: float) -> None:
    """Store the official outcome of a verify_object_lift tool call."""
    canonical_body_name = str(body_name or "").strip()
    status = "error"
    passed = False
    z_height = None

    if "PASS:" in result:
        status = "pass"
        passed = True
    elif "FAIL:" in result:
        status = "fail"

    match = re.search(r"z=([0-9.+-]+)", result)
    if match:
        try:
            z_height = float(match.group(1))
        except ValueError:
            z_height = None

    event = {
        "status": status,
        "passed": passed,
        "z_height": z_height,
        "min_height": float(min_height),
        "raw_result": result,
    }
    history = verification_history.setdefault(canonical_body_name, {"attempts": []})
    history["attempts"].append(event)


async def collect_grab_shapes_report(
    mcp_client,
    verification_history: dict,
    targets: tuple[str, ...] = GRAB_SHAPES_TARGETS,
    default_min_height: float = 0.12,
) -> dict:
    """Build a scene-grounded final report for the grab_shapes workflow."""
    objects = []
    verified_count = 0

    normalized_targets = tuple(str(body_name or "").strip() for body_name in targets)

    for body_name in normalized_targets:
        history = verification_history.get(body_name, {})
        attempts = history.get("attempts", [])
        successful_attempt = next((attempt for attempt in reversed(attempts) if attempt.get("passed")), None)
        latest_attempt = attempts[-1] if attempts else None

        try:
            result = await mcp_client.call_tool("get_object_pose", {"body_name": body_name})
            data = result.structured_content or {}
        except Exception as exc:
            data = {"status": "error", "message": str(exc)}

        if data.get("status") == "error":
            objects.append(
                {
                    "body_name": body_name,
                    "officially_picked_up": bool(successful_attempt),
                    "attempt_count": len(attempts),
                    "latest_attempt": latest_attempt,
                    "successful_attempt": successful_attempt,
                    "current_pose_error": data.get("message", "Unknown error"),
                }
            )
            if successful_attempt:
                verified_count += 1
            continue

        position = data.get("position", {})
        current_z = float(position.get("z", 0.0))
        min_height = (
            successful_attempt.get("min_height")
            if successful_attempt is not None
            else latest_attempt.get("min_height", default_min_height)
            if latest_attempt is not None
            else float(default_min_height)
        )
        currently_elevated = current_z >= min_height

        object_report = {
            "body_name": body_name,
            "officially_picked_up": bool(successful_attempt),
            "attempt_count": len(attempts),
            "latest_attempt": latest_attempt,
            "successful_attempt": successful_attempt,
            "current_pose": {
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
                "z": current_z,
            },
            "currently_elevated": currently_elevated,
            "min_height": float(min_height),
        }
        objects.append(object_report)
        if successful_attempt:
            verified_count += 1

    return {
        "workflow": "grab_shapes",
        "targets": list(normalized_targets),
        "verified_count": verified_count,
        "target_count": len(normalized_targets),
        "objects": objects,
    }


def format_grab_shapes_report(report: dict) -> str:
    """Format a grab_shapes verification report for terminal output."""
    lines = ["OFFICIAL GRAB REPORT:"]

    for obj in report.get("objects", []):
        name = obj["body_name"]
        attempt_count = obj.get("attempt_count", 0)
        successful_attempt = obj.get("successful_attempt")
        latest_attempt = obj.get("latest_attempt")

        if obj.get("current_pose_error"):
            status = "OFFICIALLY PICKED UP" if obj.get("officially_picked_up") else "NOT OFFICIALLY PICKED UP"
            lines.append(
                f"- {name}: {status}. Could not read final pose: {obj['current_pose_error']} "
                f"(verify attempts={attempt_count})."
            )
            continue

        pose = obj["current_pose"]
        if successful_attempt:
            if obj.get("currently_elevated"):
                final_state = "still elevated at the end of the run"
            else:
                final_state = "not elevated at the end of the run"
            lines.append(
                f"- {name}: OFFICIALLY PICKED UP. verify_object_lift passed "
                f"(z={successful_attempt.get('z_height', 0.0):.3f} m, threshold={obj['min_height']:.3f} m). "
                f"Final pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) and the object is {final_state}. "
                f"(verify attempts={attempt_count})"
            )
            continue

        if latest_attempt and latest_attempt.get("status") == "fail":
            verification_note = (
                f"latest verify_object_lift failed "
                f"(z={latest_attempt.get('z_height', 0.0):.3f} m, threshold={latest_attempt.get('min_height', obj['min_height']):.3f} m)"
            )
        elif latest_attempt and latest_attempt.get("status") == "error":
            verification_note = "latest verify_object_lift returned an error"
        else:
            verification_note = "verify_object_lift was never called successfully"

        lines.append(
            f"- {name}: NOT OFFICIALLY PICKED UP. {verification_note}. "
            f"Final pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}). "
            f"(verify attempts={attempt_count})"
        )

    lines.append("")
    lines.append(
        f"MISSION STATUS: Officially picked up {report.get('verified_count', 0)}/{report.get('target_count', 0)} target objects."
    )
    return "\n".join(lines)


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
    except Exception as e:
        error_msg = json.dumps({"status": "error", "message": str(e)})
        print(f"   → ❌ Failed: {error_msg}")
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
        except Exception as e:
            err_msg = (
                f"JSON PARSE ERROR: Your tool call for '{tool_name.strip()}' was malformed: {e}. "
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
