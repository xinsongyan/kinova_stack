from kinova_middleware.backend.mcp_server.toolsets.motion_tools import register_motion_tools
from kinova_middleware.backend.mcp_server.toolsets.scene_tools import register_scene_tools
from kinova_middleware.backend.mcp_server.toolsets.task_prompts import setup_prompts

__all__ = [
    "register_motion_tools",
    "register_scene_tools",
    "setup_prompts",
]
