#!/usr/bin/env python3
"""
Deepseek-driven Kinova Arm Controller for Mixed Shapes
===================================================

This script instructs an LLM to grab a box, a sphere, and a cylinder,
iteratively, using distinct geometric strategies for each.

It includes a feedback loop that verifies if the object was successfully
lifted before proceeding to the next object.
"""

import asyncio
import os
import sys
import json
from openai import OpenAI
from fastmcp import Client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_functions import load_tools_and_prompts_from_mcp, handle_prompt_tool_call, verify_lift

# Configuration
SERVER_URL = "http://127.0.0.1:8000/mcp"
# DeepSeek's free tier currently hit capacity on OpenRouter, so we fall back to a powerful free alternative.
MODEL_NAME = "openrouter/free"

TOOLS = []

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


async def main():
    # Use the OpenRouter API key provided
    api_key = os.environ.get("OPENROUTER_API_KEY", "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")

    print(f"Initializing OpenRouter client with {MODEL_NAME}...")
    # OpenRouter API behaves exactly like OpenAI but with openrouter.ai as the base url
    openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    print(f"Connecting to MCP Server at {SERVER_URL}...")
    try:
        async with Client(SERVER_URL) as mcp_client:
            print("Connected to MCP Server.")
            
            # 1. Reset scene
            await mcp_client.call_tool("reset_scene", {})
            print("Scene reset.")

            global TOOLS
            TOOLS = await load_tools_and_prompts_from_mcp(mcp_client)

            if not TOOLS:
                print("Warning: No tools loaded from MCP server.")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Grab and lift the box, sphere, and red_cylinder. Start with the red_cylinder."}
            ]

            print("\n--- Starting Mission ---\n")
            
            pending_targets = ["box", "sphere", "red_cylinder"]
            lifted_targets = []

            while True:
                try:
                    response = openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    if not hasattr(response, 'choices') or not response.choices:
                        raise ValueError(getattr(response, 'error', 'Unknown response from model provider'))
                    message = response.choices[0].message
                except Exception as api_err:
                    print(f"⚠️ Error from AI provider: {api_err}")
                    print("🔄 Informing the AI to try again in 3 seconds...")
                    messages.append({
                        "role": "user",
                        "content": f"The API failed to process your last message: {api_err}. Please correct any formatting mistakes and try again."
                    })
                    await asyncio.sleep(3)
                    continue

                messages.append(message)

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        args_str = tool_call.function.arguments
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError as json_err:
                            print(f"Error decoding arguments: {args_str}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"status": "error", "message": f"Invalid JSON format: {json_err}. Please check your syntax and try again."})
                            })
                            continue

                        print(f"🤖 AI Calling Tool: {func_name}({args})")
                        
                        # Handle prompt-related tools vs standard MCP tools
                        if func_name.startswith("get_prompt_"):
                            result_data = await handle_prompt_tool_call(mcp_client, func_name, args)
                        else:
                            try:
                                result = await mcp_client.call_tool(func_name, args)
                                result_data = result.structured_content or {}
                            except Exception as e:
                                result_data = {"status": "error", "message": f"Error executing tool: {str(e)}"}

                        if func_name == "move_pose" and result_data.get("status") == "ok":
                            for target in list(pending_targets):
                                success, current_z = await verify_lift(mcp_client, target)
                                if success:
                                    print(f"⚙️ Auto-detected lift for {target}! (Z={current_z:.3f}m)")
                                    pending_targets.remove(target)
                                    lifted_targets.append(target)
                                    alert_msg = f"System Verification: {target} was successfully lifted to z={current_z:.3f}m! Drop the object safely and proceed."
                                    if not pending_targets:
                                        alert_msg += " ALL TARGETS COMPLETED! You may now state 'All tasks complete.'"
                                    result_data["SYSTEM_ALERT"] = alert_msg

                        result_json = json.dumps(result_data, default=str)
                        # truncate output to prevent context explosion
                        if len(result_json) > 2000:
                            result_json = result_json[:2000] + "..."

                        print(f"   → Result: {result_json[:120]}{'…' if len(result_json) > 120 else ''}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_json,
                        })
                else:
                    # AI sent a chat message
                    msg_text = message.content or ""
                    print(f"\n🤖 AI Message: {msg_text}\n")
                    
                    msg_lower = msg_text.lower()
                    
                    if "all tasks complete" in msg_lower or ("complete" in msg_lower and len(pending_targets) == 0):
                        print("\n==================================")
                        print("          FINAL REPORT            ")
                        print("==================================")
                        print(f"Objects successfully lifted: {len(lifted_targets)}")
                        for k in lifted_targets:
                            print(f"  - {k}: ✓ SUCCESS")
                        break
                        
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
