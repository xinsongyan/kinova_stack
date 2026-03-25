from fastmcp import FastMCP

def setup_prompts(mcp: FastMCP):
    """Register Kinova prompts with the MCP server."""

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


    @mcp.prompt
    def stack_block(bottom_block: str, top_block: str) -> str:
        """High-level standard operating procedure for stacking top_block on bottom_block."""
        return f"""Task: Stack '{top_block}' precisely physically on top of '{bottom_block}'

Follow this step-by-step Standard Operating Procedure (SOP).
Enforce closed-loop behavior: Observe -> Act -> Verify.
CRITICAL: Execute ONE tool at a time and observe its result before proceeding.

Phase 1: Pick Up Top Block
- Use `get_object_pose` to find '{top_block}'.
- Use `move_pose` to approach '{top_block}', descend, and use `set_gripper` to grasp it safely.
- Use `move_pose` to lift '{top_block}' to a secure transit height.

Phase 2: Locate Bottom Block
- Use `get_object_pose` to query the position of the base target '{bottom_block}'.

Phase 3: Align and Descend
- Use `move_pose` to precisely align '{top_block}' directly over '{bottom_block}', maintaining a vertical gap.
- Use `move_pose` to carefully descend until '{top_block}' makes gentle contact with '{bottom_block}'.

Phase 4: Release and Retreat
- Use `set_gripper` to open the gripper entirely and release '{top_block}'. (Actuate only once).
- Use `move_pose` to move the arm straight vertically up, ensuring no lateral movement knocks over the new stack.
- Use `get_end_effector_pose` to verify a successful retreat.

Stopping Conditions & Failure Handling:
- If either block cannot be located in the workspace, abort the task altogether.
- If unexpected obstructions or unreachability occur during descent, halt immediately to avoid damaging collisions constraints.
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
