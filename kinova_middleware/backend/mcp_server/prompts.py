from fastmcp import FastMCP

def setup_prompts(mcp: FastMCP):
    """Register Kinova prompts with the MCP server."""

    @mcp.prompt
    def pick_up_block(object_name: str) -> str:
        """High-level standard operating procedure for picking up a named block."""
        return f"""Task: Pick up '{object_name}'

Follow this step-by-step Standard Operating Procedure (SOP).
Enforce closed-loop behavior: Observe -> Act -> Verify.
CRITICAL: Execute ONE tool at a time and observe its result before proceeding.

Phase 1: Observe and Locate
- Use `get_object_pose` to find the exact coordinates of '{object_name}'.
- Verify the object is detected and within the reachable workspace.

Phase 2: Pre-Grasp Approach
- Use `move_pose` to navigate the end-effector to a safe standoff distance right above '{object_name}'.
- Verify the arm has reached the pre-grasp pose securely.

Phase 3: Grasp Execution
- Use `move_pose` to carefully descend to the object's grasping height.
- Use `set_gripper` to close the gripper fingers around the object.
- CRITICAL: Execute `set_gripper` exactly once. Do not repeat gripper commands.

Phase 4: Lift and Verify
- Use `move_pose` to lift the object straight upward.
- Use `get_joint_state` and `get_end_effector_pose` to verify the arm holds the elevated position.

Stopping Conditions & Failure Handling:
- If `get_object_pose` fails to locate the object, abort the sequence immediately.
- If a movement command fails, do not forcefully retry the same path blindly. Stop and report the failure.
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
