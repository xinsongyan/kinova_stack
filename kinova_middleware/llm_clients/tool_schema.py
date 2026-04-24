from __future__ import annotations

import json


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
    except Exception as exc:
        print(f"Failed to load tools and prompts: {exc}")
        return []


def bind_model_tools(
    llm,
    tools_schema: list[dict],
    tool_choice: str | None = "required",
    parallel_tool_calls: bool | None = None,
):
    """Bind tools to an LLM with a provider-compatible tool choice mode."""
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
