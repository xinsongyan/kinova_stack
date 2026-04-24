from __future__ import annotations

import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCENES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "scenes"))


def discover_local_scene_names() -> list[str]:
    """Return the locally available scene basenames in server selection order."""
    if not os.path.isdir(_SCENES_DIR):
        return []
    return [
        name
        for name in sorted(os.listdir(_SCENES_DIR))
        if name.endswith((".xml", ".mjcf"))
    ]


def resolve_local_scene_number(scene_name: str) -> int:
    """Resolve a scene basename to the 1-based number used by reset_scene()."""
    normalized_name = os.path.basename(scene_name)
    scene_names = discover_local_scene_names()
    if normalized_name not in scene_names:
        raise ValueError(
            f"Scene '{normalized_name}' was not found in {_SCENES_DIR}. "
            f"Available scenes: {', '.join(scene_names) if scene_names else 'none'}."
        )
    return scene_names.index(normalized_name) + 1


async def reset_scene_if_available(
    mcp_client,
    available_tool_names: set[str],
    *,
    scene_name: str | None = None,
    scene_number: int | None = None,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> None:
    """Reset or hot-swap the scene before an agent run when supported."""
    del total_runs
    if "reset_scene" not in available_tool_names:
        print("Warning: reset_scene() is not available, so the scene will not auto-reset.")
        return

    selected_scene_number = scene_number
    if selected_scene_number is None and scene_name is not None:
        selected_scene_number = resolve_local_scene_number(scene_name)

    tool_args = {}
    if selected_scene_number is not None:
        tool_args["scene_number"] = selected_scene_number

    result = await mcp_client.call_tool("reset_scene", tool_args)
    data = result.structured_content or {}
    status = data.get("status", "unknown")
    message = data.get("message", "No reset_scene message returned.")

    prefix = f"[Run {run_number}] " if run_number is not None else ""
    print(f"{prefix}reset_scene -> {status}: {message}")
