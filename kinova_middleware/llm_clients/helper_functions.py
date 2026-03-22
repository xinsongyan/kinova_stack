import json

async def load_tools_and_prompts_from_mcp(mcp_client):
    """
    Loads tools from the MCP server.
    Also fetches available prompts and wraps them as callable OpenAI tools,
    allowing the LLM to query Standard Operating Procedures (SOPs) on demand.
    """
    print("Fetching tools from MCP server...")
    try:
        mcp_tools = await mcp_client.list_tools()
        openai_tools = []
        for tool in mcp_tools:
            function_def = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {}
            }
            openai_tools.append({
                "type": "function",
                "function": function_def
            })
            print(f" - Loaded tool: {tool.name}")
            
        print("Fetching prompts from MCP server...")
        mcp_prompts = await mcp_client.list_prompts()
        
        for prompt in mcp_prompts:
            print(f" - Loaded prompt template as tool: get_prompt_{prompt.name}")
            
            # Create a tool definition that lets the AI fetch this prompt
            properties = {}
            required = []
            if prompt.arguments:
                for arg in prompt.arguments:
                    properties[arg.name] = {
                        "type": "string",
                        "description": arg.description or f"The {arg.name} to insert into the prompt"
                    }
                    if arg.required:
                        required.append(arg.name)

            prompt_tool_def = {
                "name": f"get_prompt_{prompt.name}",
                "description": prompt.description or f"Get the Standard Operating Procedure (SOP) for {prompt.name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            openai_tools.append({
                "type": "function",
                "function": prompt_tool_def
            })

        return openai_tools
    except Exception as e:
        print(f"Failed to load tools and prompts: {e}")
        return []

async def handle_prompt_tool_call(mcp_client, func_name, args):
    """
    If the AI calls a tool starting with 'get_prompt_', we route it to get_prompt.
    """
    prompt_name = func_name.replace("get_prompt_", "", 1)
    try:
        result = await mcp_client.get_prompt(prompt_name, args)
        if result and hasattr(result, "messages") and result.messages:
            content = result.messages[0].content
            return {"status": "ok", "prompt_text": content.text if hasattr(content, "text") else str(content)}
        return {"status": "error", "message": "Prompt returned empty messages."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get prompt: {e}"}

async def verify_lift(mcp_client, body_name):
    """Checks if the object's Z height is significantly above the board (e.g. > 0.10m)."""
    try:
        result = await mcp_client.call_tool("get_object_pose", {"body_name": body_name})
        data = result.structured_content or {}
        if data.get("status") == "error":
            return False, 0.0
        z = data.get("position", {}).get("z", 0.0)
        return z > 0.12, z
    except Exception as e:
        print(f"Verification error: {e}")
        return False, 0.0
