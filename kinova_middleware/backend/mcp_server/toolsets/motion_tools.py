from __future__ import annotations

from collections.abc import Callable

from fastmcp import FastMCP

from kinova_middleware.backend.interfaces.capabilities import BackendCapability
from kinova_middleware.backend.mcp_server.services import KinovaMotionToolService


def register_motion_tools(
    mcp: FastMCP,
    *,
    motion_service: KinovaMotionToolService,
    has_capability: Callable[[BackendCapability], bool],
) -> list[str]:
    registered_tools: list[str] = []

    def register_tool_if(enabled: bool):
        def decorator(fn):
            if enabled:
                registered_tools.append(fn.__name__)
                return mcp.tool()(fn)
            return fn

        return decorator

    @register_tool_if(has_capability(BackendCapability.ARM_MOTION))
    def move_home() -> dict:
        return motion_service.move_home()

    @register_tool_if(has_capability(BackendCapability.ARM_MOTION))
    def get_end_effector_pose() -> dict:
        return motion_service.get_end_effector_pose()

    @register_tool_if(has_capability(BackendCapability.GRIPPER_CONTROL))
    def set_gripper(percent: float) -> dict:
        return motion_service.set_gripper(percent)

    @register_tool_if(
        has_capability(BackendCapability.ARM_MOTION)
        and has_capability(BackendCapability.IK_SOLVER)
    )
    def move_pose(
        target_pos: list[float] | str,
        target_quat: list[float] | str,
        seed_q_rad: list[float] | str | None = None,
        allow_orientation_fallback: bool = True,
        move_wrist: bool = True,
    ) -> dict:
        return motion_service.move_pose(
            target_pos,
            target_quat,
            seed_q_rad,
            allow_orientation_fallback=allow_orientation_fallback,
            move_wrist=move_wrist,
        )

    @register_tool_if(has_capability(BackendCapability.ARM_MOTION))
    def rotate_wrist(angle_deg: float) -> dict:
        return motion_service.rotate_wrist(angle_deg)

    return registered_tools
