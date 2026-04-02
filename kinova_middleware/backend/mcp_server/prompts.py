from fastmcp import FastMCP

def setup_prompts(mcp: FastMCP):
    """Register Kinova prompts with the MCP server."""

    @mcp.prompt
    def grab_shapes() -> str:
        """Task prompt for grabbing and lifting the demo shapes."""
        return """You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.
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
   CRITICAL: Use `target_quat=[0.0, 0.0, 0.0, 0.0]` and pass the argument `"move_wrist": False` to prevent IK orientation failures while lifting straight up.
9. **Verify Lift**: If the local client tool `verify_object_lift(body_name='<target>')` is available, call it after the lift and use it to confirm the object is high enough before moving on.

Once you receive confirmation for all three objects (box, sphere, red_cylinder), state "All tasks complete."

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""


    @mcp.prompt
    def sort_cubes() -> str:
        """Task prompt for sorting cubes into their bins."""
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
   a. Identify its color and target bin.
   b. Execute a precise 7-step pick-and-place sequence (see below).
3. Finish when `check_sorting_status()` reports "MISSION STATUS: ALL CUBES SORTED SUCCESSFULLY".

Detailed 7-Step Pick Sequence (REQUIRED):
1. `get_object_pose(body_name='...')` and `compute_grasp_height(...)`.
2. `move_pose` to `[x, y, top_height + 0.10]` with `target_quat=[0,0,0,0]` (Approach).
3. `get_end_effector_pose()` and `compute_wrist_alignment(...)` to find rotation.
4. `rotate_wrist(angle_deg)` to align the fingers.
5. `move_pose` to `[x, y, top_height + 0.015]` with `target_quat=[0,0,0,0]` (Descend).
6. `set_gripper(percent=0.54)` (Grasp).
7. `move_pose` to `[x, y, top_height + 0.20]` with `move_wrist=False` (Lift).

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


    @mcp.prompt
    def stack_cubes(bottom_block: str, top_block: str) -> str:
        """Task prompt for stacking top_block on bottom_block using the stack_cubes workflow."""
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
3. get_object_pose(all) (this will return the pose of all objects)
4. compute all required pick/place coordinates in advance
5. Compute Top Height: Call `compute_grasp_height` using the object's geom_type, size, and quaternion to find `top_height`, the physical top boundary.
6. Move to top pregrasp: Call `move_pose` to the object's x,y, and Z = top_height + 0.15. Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
7. Align Wrist:
     a. Call `get_end_effector_pose()` to get the current arm orientation (EE quat).
     b. Call `compute_wrist_alignment` passing the object's quaternion and the EE quaternion.
     c. If box: the result angle might need modulus math to snap to a 90-degree face. Usually just apply the raw angle unless it's way off. (Hint: angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0)
     d. Call `rotate_wrist(angle_deg)` with the final alignment angle.
8. Descend:
   Call `move_pose` to the object's x,y, and Z = top_height + 0.02.
   Use `target_quat=[0.0, 0.0, 0.0, 0.0]`.
9. Close gripper (0.58 is a good value for cubes)
10. Immediately lift to the precomputed lift point
11. **Prelocate Stack Target**: Call `get_object_pose(body_name='{bottom_block}')`. Note its position and half-height (size[2]).
12. **Precalculate Stacking Height**: Determine the center position for '{top_block}': `target_z = bottom_z + bottom_hh + top_hh + 0.01` (includes a small safety margin).
13. **Pre-Place**:
    a. Call `move_pose` to `[bottom_x, bottom_y, target_z + 0.10]` with `target_quat=[0.0, 0.0, 0.0, 0.0]`.

14. **Place Descend**:
    Call `move_pose` to the corrected X,Y and `Z = target_z + 0.04`.
    If you get ik_failed that means you may have hit the base cube, so get '{top_block}' current pose and inspect whether its z is already consistent with a successful stack before releasing.
15. **Release and Retreat**:
    a. Call `set_gripper(percent=1.0)`.
    b. Call `move_pose` straight up to `target_z + 0.15` with `"move_wrist": False`.
    c. Call `move_home()`.
16. **Verify Stack**:
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


    @mcp.prompt
    def pick_up_block(object_name: str) -> str:
        """High-level standard operating procedure for picking up a named block."""
        return f"""Task: Pick up '{object_name}'

You are a robotic arm control agent. You control a 4-DOF Kinova Gen3 arm with a 2-finger gripper.

STANDARD OPERATING PROCEDURE FOR ANY OBJECT:
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
9. **Verify Lift**: Confirm the object is held (check for SYSTEM_ALERT in tool output or re-query object Z position).

Output policy:
- Prefer tool calls over chat.
- Execute steps sequentially and rely on tool outputs rather than guessing values.
- there is a z limit at 0.07 do not set the arm to go lower than this
"""


    @mcp.prompt
    def place_block(location_name: str) -> str:
        """High-level standard operating procedure for placing a held block."""
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

#     @mcp.prompt
#     def recover_failed_grasp(object_name: str) -> str:
#         """High-level standard operating procedure for recovering from a missed grasp."""
#         return f"""Task: Recover from a failed grasp attempting to hold '{object_name}'

# Follow this step-by-step Standard Operating Procedure (SOP).
# Enforce closed-loop behavior: Observe -> Act -> Verify.
# CRITICAL: Execute ONE tool at a time and observe its result before proceeding.

# Phase 1: Diagnose and Clear
# - Use `get_joint_state` and `get_end_effector_pose` to understand the robot's current posture and verify safety.
# - Use `set_gripper` to fully open the gripper, clearing any snagged states. (Actuate once).

# Phase 2: Reset and Reposition
# - Use `move_joints` or `move_pose` to safely retract the arm to a high, neutral standby position.
# - Use `get_object_pose` to recalculate the exact pose of '{object_name}'. It may have been nudged during the failed attempt.

# Phase 3: Approach and Retry
# - Driven by the freshly observed pose, use `move_pose` to approach the object from an optimized angle and clearance.
# - Use `move_pose` to descend accurately to the corrected grasping plane.
# - Use `set_gripper` to close the fingers and enforce the grasp again.

# Phase 4: Verification
# - Use `move_pose` to attempt lifting the block slightly.
# - Verify stability and success using `get_joint_state` parameters.

# Stopping Conditions & Failure Handling:
# - Explicitly avoid infinite blind retry loops. Hard limit this recovery to ONE exact attempt.
# - If the object was pushed out of range, or if the secondary attempt fails identically, halt all operations and report a critical failure.
# """
