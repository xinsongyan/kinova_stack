from __future__ import annotations

from fastmcp import FastMCP

from kinova_middleware.backend.interfaces.capabilities import BackendCapability


def setup_prompts(
    mcp: FastMCP,
    *,
    capabilities: frozenset[BackendCapability] | None = None,
) -> list[str]:
    register_all = capabilities is None
    has_object_query = register_all or BackendCapability.OBJECT_QUERY in capabilities
    registered_prompts: list[str] = []

    def register_prompt_if(enabled: bool):
        def decorator(fn):
            if enabled:
                registered_prompts.append(fn.__name__)
                return mcp.prompt(fn)
            return fn

        return decorator

    @register_prompt_if(has_object_query)
    def grab_shapes() -> str:
        return """You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.
Your mission is to sequentially pick up three objects in the scene: "box", "sphere", and "cylinder".

STANDARD OPERATING PROCEDURE FOR EACH OBJECT:
1. **Home & Prepare**: Call `move_home()`. Call `set_gripper(percent=1.0)` to open fingers.
2. **Plan The Pick**: Call `plan_object_grasp(body_name='<target>', profile='shapes')`.
   Use the returned `approach_move`, `grasp_move`, `lift_move`, and `recommended_gripper_percent`.
3. **Approach**: Call `move_pose` using the returned `approach_move`.
4. **Align Wrist**:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `plan_wrist_alignment(body_name='<target>', ee_quat_xyzw=[...])`.
     c. Call `rotate_wrist(angle_deg)` using the returned final angle.
5. **Descend**:
   Call `move_pose` using the returned `grasp_move`.
6. **Grasp**:
   Call `set_gripper(percent=...)` using `recommended_gripper_percent` from `plan_object_grasp`.
7. **Lift Safely**: Call `move_pose` using the returned `lift_move`.
8. **Verify Lift**: If the local client tool `verify_object_lift(body_name='<target>')` is available, call it after the lift and use it to confirm the object is high enough before moving on.
9. **Drop the object**: Call `set_gripper(percent=1.0)` to release the object.

Once you receive confirmation for all three objects (box, sphere, cylinder), state "All tasks complete."

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- Before every tool call, first say one short sentence explaining what you are about to do and why.
- Example: "I am going to get the position of the cylinder so I can plan the grasp."
- After that one sentence, immediately make the tool call.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""

    @register_prompt_if(has_object_query)
    def sort_cubes() -> str:
        return """You are controlling a Kinova robot through MCP tools to perform a sorting task.
Your mission is to sort all red cubes into the red bin ("red_bin_target") and all blue cubes into the blue bin ("blue_bin_target").

CRITICAL TOOL RULES:
- **ONLY** use tools that are officially provided in the tool schema.
- **NEVER** invent or use a tool called `pick_cube` or `place_cube`. They DO NOT exist.
- You MUST execute the full 7-step pick sequence manually for every object.
- **ONLY** use the structured tool-calling format (no raw `<|tool_calls|>` text in the response).

Mission objective:
1. Use `check_sorting_status()` to identify the status of all cubes.
2. For each cube marked as "At Starting Position":
   a. Call `plan_object_grasp(body_name='<cube>', profile='sort_cubes')`.
   b. Execute the required pick sequence using the returned moves and gripper percent.
   c. Call `plan_bin_place(body_name='<cube>', profile='sort_cubes')`.
   d. Use the returned placement moves to place it in the correct bin.
3. Finish when `check_sorting_status()` reports "MISSION STATUS: ALL CUBES SORTED SUCCESSFULLY".

Detailed 7-Step Pick Sequence (REQUIRED):
1. `plan_object_grasp(body_name='...', profile='sort_cubes')`.
2. `move_pose` using the returned `approach_move`.
3. `get_end_effector_pose()` and `plan_wrist_alignment(...)` to find the final wrist rotation.
4. `rotate_wrist(angle_deg)` to align the fingers.
5. `move_pose` using the returned `grasp_move`.
6. `set_gripper(percent=recommended_gripper_percent)` from the grasp plan.
7. `move_pose` using the returned `lift_move`.

Required Bin Placement Sequence:
1. Call `plan_bin_place(body_name='<cube>', profile='sort_cubes')`.
2. Call `move_pose` using the returned `place_move`.
3. Call `set_gripper(percent=1.0)` to release the cube into the bin.
4. Call `move_pose` using the returned `retreat_move`.

Safety and execution rules:
- NEVER attempt to pick up any cube marked as "OUT OF WORKSPACE".
- Focus strictly on cubes "At Starting Position".
- never go above 0.24m in the z axis when pregrasping.
- dont overthink ik error messages if the error is a couple of cm or less.

Output style:
- Briefly acknowledge the current state.
- Proceed to the NEXT tool call immediately.
- DO NOT summarize the whole task at once.
"""

    @register_prompt_if(has_object_query)
    def stack_cubes(bottom_block: str, top_block: str) -> str:
        return f"""Task: Stack '{top_block}' directly above '{bottom_block}'

You are controlling a Kinova robot through MCP tools. Your task is to stack '{top_block}' directly above '{bottom_block}'.

Mission objective:
1. Identify the pose of '{bottom_block}' and '{top_block}'.
2. Pick up '{top_block}'.
3. Place '{top_block}' centered on top of '{bottom_block}'.
4. Finish safely.

Critical execution rule:
Before you close the gripper on '{top_block}', you MUST calculate and store all pick-and-place target coordinates first, so the robot does not waste time while holding the cube and risk the cube slipping out.

Required planning order:
1. Query the '{top_block}' pose.
2. Query the '{bottom_block}' pose.
3. Determine the '{top_block}' size and '{bottom_block}' size if available.
4. Compute and mentally lock in these target positions BEFORE grasping:
   - top pregrasp position
   - top grasp position
   - top lift position
   - target preplace position above '{bottom_block}'
   - target place position on top of '{bottom_block}'
   - target retreat position
5. Only after all of the above are known, execute the pick.
6. After grasping, move immediately through lift -> preplace -> place -> release -> retreat with no unnecessary pauses.

Tool-use constraints:
- Use only available MCP tools.
- Do not invent robot states or object poses.
- Always query real poses from tools.
- Do not delay after grasping except for the minimum needed to confirm the grasp.
- Do not recalculate placement while the cube is already in hand unless absolutely necessary.
- Keep motions efficient and direct.
- Prefer position-only IK when orientation is not essential, but align the wrist if needed for a stable grasp/place.
- If grasp verification fails, safely reopen the gripper, retreat, and try again logically.

Placement requirements:
- '{top_block}' must be centered over '{bottom_block}' in x and y.
- The place height must correspond to the top surface of '{bottom_block}' plus '{top_block}' half-height, with a small safe margin only if needed.
- The final stack should be vertical and stable.

Recommended execution sequence:
1. move_home
2. open gripper
3. Call `plan_object_grasp(body_name='{top_block}', profile='stack_cubes')`.
4. Call `plan_stack_place(bottom_block='{bottom_block}', top_block='{top_block}', profile='stack_cubes')` BEFORE grasping.
5. Move to top pregrasp: Call `move_pose` using the returned `approach_move`.
6. Align Wrist:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `plan_wrist_alignment(body_name='{top_block}', ee_quat_xyzw=[...])`.
     c. Call `rotate_wrist(angle_deg)` with the returned final alignment angle.
7. Descend:
   Call `move_pose` using the returned `grasp_move`.
8. Close gripper using `recommended_gripper_percent` from the grasp plan.
9. Immediately lift using the returned `lift_move`.
10. **Pre-Place**:
    a. Call `move_pose` using the returned `preplace_move` from `plan_stack_place`.

11. **Place Descend**:
    Call `move_pose` using the returned `place_move` from `plan_stack_place`.
    If you get ik_failed that means you may have hit the base cube, so get '{top_block}' current pose and inspect whether its z is already consistent with a successful stack before releasing.
12. **Release and Retreat**:
    a. Call `set_gripper(percent=1.0)`.
    b. Call `move_pose` using the returned `retreat_move` from `plan_stack_place`.
    c. Call `move_home()`.
13. **Verify Stack**:
    If the local client tool `check_stacking_status()` is available, call it to confirm which cube is on top of which cube after release.

Reasoning rules:
- Think step by step.
- Be concise internally.
- Do not spend time over-planning after the grasp.
- The key requirement is: precompute lift, preplace, and place coordinates BEFORE picking up '{top_block}'.
- never go above 0.24m in the z axis

Output style:
- Briefly state what you are doing before each major action.
- Report important pose values and errors when useful.
- If a step fails, explain why and recover safely.
"""

    @register_prompt_if(has_object_query)
    def pick_up_block(object_name: str) -> str:
        return f"""Task: Pick up '{object_name}'

You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.

STANDARD OPERATING PROCEDURE FOR ANY OBJECT:
1. **Home & Prepare**: Call `move_home()`. Call `set_gripper(percent=1.0)` to open fingers.
2. **Plan The Pick**: Call `plan_object_grasp(body_name='{object_name}', profile='generic')`.
   Use the returned `approach_move`, `grasp_move`, `lift_move`, and `recommended_gripper_percent`.
3. **Approach**: Call `move_pose` using the returned `approach_move`.
4. **Align Wrist**:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `plan_wrist_alignment(body_name='{object_name}', ee_quat_xyzw=[...])`.
     c. Call `rotate_wrist(angle_deg)` with the returned final alignment angle.
5. **Descend**: 
   Call `move_pose` using the returned `grasp_move`.
6. **Grasp**: 
   Call `set_gripper(percent=...)` using `recommended_gripper_percent` from `plan_object_grasp`.
7. **Lift Safely**: Call `move_pose` using the returned `lift_move`.
8. **Verify Lift**: Confirm the object is held (check for SYSTEM_ALERT in tool output or re-query object Z position).

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""

    @register_prompt_if(has_object_query)
    def place_block(location_name: str) -> str:
        return f"""Task: Place the currently held block at '{location_name}'

Follow this step-by-step Standard Operating Procedure (SOP).
Enforce closed-loop behavior: Observe -> Act -> Verify.
CRITICAL: Execute ONE tool at a time and observe its result before proceeding.

Phase 1: Locate Destination
- Use `get_object_pose` to determine the exact coordinates of the target place '{location_name}'.

Phase 2: Pre-Place Approach
- Use `move_pose` to move the arm (while safely holding the block) to a clearance height directly above '{location_name}'.

Phase 3: Place Action
- Use `move_pose` to slowly descend to the designated placement level.
- Use `set_gripper` to open the gripper and release the block.
- CRITICAL: Execute the release command exactly once to prevent gripper stutter.

Phase 4: Retreat and Verify
- Use `move_pose` to retreat the empty open gripper straight up to a stable travel height.
- Use `get_end_effector_pose` to confirm the arm cleanly cleared the placement area.

Stopping Conditions & Failure Handling:
- If the drop location cannot be verified, keep holding the block and stop the procedure.
- If the arm cannot safely reach the placement pose, do not drop the object blindly. Stop and report the error.
"""

    return registered_prompts
