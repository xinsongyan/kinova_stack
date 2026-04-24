from __future__ import annotations

from collections.abc import Callable

from fastmcp import FastMCP

from kinova_middleware.backend.interfaces.capabilities import BackendCapability
from kinova_middleware.backend.mcp_server.services import (
    GeometryToolService,
    KinovaMotionToolService,
    TaskPlanningToolService,
)


def register_scene_tools(
    mcp: FastMCP,
    *,
    motion_service: KinovaMotionToolService,
    geometry_service: GeometryToolService,
    task_planning_service: TaskPlanningToolService,
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

    @register_tool_if(has_capability(BackendCapability.SCENE_CONTROL))
    def reset_scene(scene_number: int | str | None = None) -> dict:
        return motion_service.reset_scene(scene_number)

    @register_tool_if(has_capability(BackendCapability.OBJECT_QUERY))
    def get_object_pose(body_name: str) -> dict:
        return motion_service.get_object_pose(body_name)

    @register_tool_if(has_capability(BackendCapability.OBJECT_QUERY))
    def plan_object_grasp(body_name: str, profile: str = "shapes") -> dict:
        return task_planning_service.plan_object_grasp(body_name, profile=profile)

    @register_tool_if(has_capability(BackendCapability.OBJECT_QUERY))
    def plan_wrist_alignment(body_name: str, ee_quat_xyzw: list[float] | str) -> dict:
        return task_planning_service.plan_wrist_alignment(body_name, ee_quat_xyzw)

    @register_tool_if(has_capability(BackendCapability.OBJECT_QUERY))
    def plan_bin_place(
        body_name: str,
        target_name: str | None = None,
        profile: str = "sort_cubes",
    ) -> dict:
        return task_planning_service.plan_bin_place(body_name, target_name=target_name, profile=profile)

    @register_tool_if(has_capability(BackendCapability.OBJECT_QUERY))
    def plan_stack_place(
        bottom_block: str,
        top_block: str,
        profile: str = "stack_cubes",
    ) -> dict:
        return task_planning_service.plan_stack_place(bottom_block, top_block, profile=profile)

    @register_tool_if(True)
    def compute_grasp_height(
        geom_type: str,
        size: list[float] | str,
        quat_xyzw: list[float] | str,
    ) -> dict:
        return geometry_service.compute_grasp_height(geom_type, size, quat_xyzw)

    @register_tool_if(True)
    def compute_wrist_alignment(
        obj_quat_xyzw: list[float] | str,
        ee_quat_xyzw: list[float] | str,
    ) -> dict:
        return geometry_service.compute_wrist_alignment(obj_quat_xyzw, ee_quat_xyzw)

    return registered_tools
