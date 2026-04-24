from __future__ import annotations

import logging

from fastmcp import FastMCP

from kinova_middleware.backend.interfaces.capabilities import (
    BackendCapability,
    get_supported_capabilities,
)
from kinova_middleware.backend.mcp_server.services import (
    GeometryToolService,
    KinovaMotionToolService,
    TaskPlanningToolService,
    ToolRuntimeContext,
)
from kinova_middleware.backend.mcp_server.toolsets.motion_tools import register_motion_tools
from kinova_middleware.backend.mcp_server.toolsets.scene_tools import register_scene_tools


log = logging.getLogger("mcp_kinova")


def _resolve_capabilities(state: dict) -> frozenset[BackendCapability] | None:
    explicit = state.get("capabilities")
    if explicit is not None:
        return frozenset(BackendCapability(cap) for cap in explicit)

    get_controller = state.get("get_controller")
    if not callable(get_controller):
        return None

    try:
        controller = get_controller()
    except Exception:
        return None

    if not hasattr(controller, "supported_capabilities"):
        return None
    return get_supported_capabilities(controller)


def setup_tools(mcp: FastMCP, state: dict) -> list[str]:
    """Register Kinova tools with the MCP server."""

    capabilities = _resolve_capabilities(state)
    register_all = capabilities is None

    runtime = ToolRuntimeContext(
        get_controller=state["get_controller"],
        motion_lock=state["motion_lock"],
        physics_lock=state["physics_lock"],
        run_until_reached=state["run_until_reached"],
        reset_or_reload_scene=state.get("reset_or_reload_scene"),
    )
    motion_service = KinovaMotionToolService(runtime, logger=log)
    geometry_service = GeometryToolService()
    task_planning_service = TaskPlanningToolService(runtime, geometry_service, logger=log)

    def has_capability(capability: BackendCapability) -> bool:
        return register_all or capability in capabilities

    registered_tools = []
    registered_tools.extend(
        register_motion_tools(
            mcp,
            motion_service=motion_service,
            has_capability=has_capability,
        )
    )
    registered_tools.extend(
        register_scene_tools(
            mcp,
            motion_service=motion_service,
            geometry_service=geometry_service,
            task_planning_service=task_planning_service,
            has_capability=has_capability,
        )
    )
    return registered_tools
