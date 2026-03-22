#!/usr/bin/env python3
"""
OpenAI-driven Kinova Arm Controller for Mixed Shapes
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
MODEL_NAME = "gpt-4o-mini"

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
9. **Declare Success**: Once the lift command finishes, you MUST send a plain text message containing EXACTLY the phrase:
   "<target> lifted"
   For example, if you just lifted the box, say: "box lifted".

The system will then verify your work. If it's true, it will tell you. If it's false, it will tell you. In either case, it will then instruct you to move on to the next object.

Once you have verified all three objects (box, sphere, red_cylinder), state "All tasks complete."

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Initializing OpenAI client...")
    openai_client = OpenAI(api_key=api_key)

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
            
            results_log = {}

            while True:
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

                message = response.choices[0].message
                messages.append(message)

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        args_str = tool_call.function.arguments
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            print(f"Error decoding arguments: {args_str}")
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
                    
                    found_target = None
                    for target in ["box", "sphere", "red_cylinder"]:
                        if f"{target} lifted" in msg_lower:
                            found_target = target
                            break
                            
                    if found_target:
                        print(f"⚙️ Verifying lift for {found_target}...")
                        success, current_z = await verify_lift(mcp_client, found_target)
                        
                        results_log[found_target] = success
                        status_str = "SUCCESS" if success else "FAILED"
                        print(f"   → {found_target} verification: {status_str} (Z={current_z:.3f}m)")
                        
                        # Tell AI
                        sys_msg = f"System Verification for {found_target}: {status_str} (z={current_z:.3f}m). Drop the object and proceed to the next object. If you have done all 3, declare 'All tasks complete.'"
                        messages.append({"role": "user", "content": sys_msg})
                        print(f"   → Feeding back to AI: {sys_msg}")
                        continue
                        
                    if "all tasks complete" in msg_lower or ("complete" in msg_lower and len(results_log) == 3):
                        print("\n==================================")
                        print("          FINAL REPORT            ")
                        print("==================================")
                        success_count = sum(1 for v in results_log.values() if v)
                        print(f"Objects successfully lifted: {success_count} / {len(results_log)}")
                        for k, v in results_log.items():
                            print(f"  - {k}: {'✓ SUCCESS' if v else '✗ FAILED'}")
                        break
                        
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
