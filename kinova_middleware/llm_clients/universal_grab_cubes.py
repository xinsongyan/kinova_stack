#!/usr/bin/env python3
"""
Universal FastMCP Agent
=======================

This script acts as a completely agnostic, universal LLM agent that connects to ANY 
FastMCP server using `fastmcp.Client`. It has ZERO hardcoded logic about robot arms, 
shapes, or specific tools!

It uses LangChain to dynamically load whatever tools and Prompts the server happens 
to provide, and sets them up for `gpt-4o-mini` (or any model you choose).

Usage:
  python universal_fastmcp_agent.py --url "http://127.0.0.1:8000/mcp" --task "Clean up the blocks on the table"
"""

import asyncio
import os
import sys
import json
import argparse
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

# =========================================================================
# CONFIGURATION
# -> Modify these variables to easily switch between models and APIs!
# =========================================================================
MODEL_NAME = "deepseek-ai/deepseek-v3.2"
BASE_URL = "https://integrate.api.nvidia.com/v1"


SYSTEM_PROMPT = """
You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.
Your mission is to sequentially pick up three objects in the scene: "box", "sphere", and "red_cylinder".

STANDARD OPERATING PROCEDURE FOR EACH OBJECT:
1. **Home & Prepare**: Call `move_home()`. Call `set_gripper(percent=1.0)` to open fingers.
2. **Locate Object**: Call `get_object_pose(body_name='<target>')`. Read its position, size, geom_type, and quaternion.
   - For box, geom_type="box". 
   - For sphere, geom_type="sphere". 
   - For red_cylinder, geom_type="cylinder".
3. **Compute Top Height**: Call `compute_grasp_height` using the object's geom_type, size, and quaternion to find `top_height`, the physical top boundary.
4. **Approach**: Call `move_pose` to the object's x,y, and Z = top_height + 0.15. Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
5. **Align Wrist**:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `compute_wrist_alignment` passing the object's quaternion and the EE quaternion.
     c. If box: the result angle might need modulus math to snap to a 90-degree face. Usually just apply the raw angle unless it's way off. (Hint: angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0)
     d. Call `rotate_wrist(angle_deg)` with the final alignment angle.
6. **Descend**: 
   Call `move_pose` to the object's x,y, and Z = top_height + 0.01.
   Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
7. **Grasp**: 
   Call `set_gripper(percent=...)`:
   - **sphere**: percent=0.62
   - **cylinder**: percent=0.55
   - **box**: percent=0.58
8. **Lift Safely**: Call `move_pose` using the current x,y, and Z = top_height + 0.20. 
   CRITICAL: Use `target_quat=[0.0, 0.0, 0.0, 0.0]` and pass the argument `"move_wrist": False` to prevent IK orientation failures while lifting straight up!
9. **Verify Lift**: After you call `move_pose` to lift the object safely, the corresponding tool output will automatically include a `"SYSTEM_ALERT"` letting you know if the object was successfully detected as lifted.
   Once you receive the success alert inside the tool outputs, drop the object safely and proceed.

Once you receive confirmation for all three objects (box, sphere, red_cylinder), state "All tasks complete."

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""

async def execute_mcp_tool(mcp_client, tool_name: str, args: dict) -> str:
    """Executes an MCP tool dynamically by name and returns a stringified JSON representation."""
    print(f"\n🚀 Agent Executing: {tool_name}({args})")
    
    # If the agent is trying to query a prompt (because we wrapped them as tools)
    if tool_name.startswith("get_prompt_"):
        prompt_name = tool_name.replace("get_prompt_", "", 1)
        try:
            result = await mcp_client.get_prompt(prompt_name, args)
            text = result.messages[0].content.text if hasattr(result.messages[0].content, "text") else str(result.messages[0].content)
            print(f"   → Read SOP Prompt: {prompt_name}")
            return text
        except Exception as e:
            return json.dumps({"error": f"Failed to get prompt: {str(e)}"})

    # Otherwise, it is a standard MCP tool call
    try:
        result = await mcp_client.call_tool(tool_name, args)
        result_json = json.dumps(result.structured_content or {}, default=str)
        # Prevent context explosion on giant return values
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "... [truncated]"
        print(f"   → Success: {result_json[:200]}...")
        return result_json
    except Exception as e:
        error_msg = json.dumps({"status": "error", "message": str(e)})
        print(f"   → ❌ Failed: {error_msg}")
        return error_msg

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
            
            # 1. Dynamically load native tools
            mcp_tools = await mcp_client.list_tools()
            tools_schema = []
            for tool in mcp_tools:
                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {}
                    }
                })
            
            # 2. Dynamically load server Prompts, wrap them as tools!
            mcp_prompts = await mcp_client.list_prompts()
            for prompt in mcp_prompts:
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

                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": f"get_prompt_{prompt.name}",
                        "description": prompt.description or f"Query the SOP/instructions for {prompt.name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                })
                
            print(f"Successfully loaded {len(mcp_tools)} tools and {len(mcp_prompts)} dynamic prompts!")
            
            # 3. Bind the extracted tools to our Langchain Agent
            llm_with_tools = llm.bind_tools([t["function"] for t in tools_schema])
            
            # Injecting a universal identity hint
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="Grab and lift the box, sphere, and red_cylinder. Start with the red_cylinder.")
            ]
            
            # Infinite Action Loop (stops when LLM gives up or finishes)
            iteration = 1
            while True:
                print(f"--- [Thinking - Step {iteration}] ---")
                
                # Predict next action
                ai_msg = llm_with_tools.invoke(messages)
                messages.append(ai_msg)
                
                # Check if it wants to use tools
                if ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        result_str = await execute_mcp_tool(
                            mcp_client, 
                            tool_call["name"], 
                            tool_call["args"]
                        )
                        # Return the result back into the Agent's context
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"], 
                                name=tool_call["name"], 
                                content=result_str
                            )
                        )
                else:
                    # No tool calls, the AI just gave a conversational response (mission complete or clarifying question)
                    print(f"\n🤖 Agent Final Report: {ai_msg.content}")
                    break
                    
                iteration += 1

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
