"""Scene selector for Kinova MuJoCo simulation.

Scans the ``scenes/`` directory for ``.xml`` / ``.mjcf`` files, presents an
interactive menu, and returns the chosen file path.

Usage:
    from scene_selector import select_scene
    scene_path = select_scene()          # interactive menu
    scene_path = select_scene()          # or honour KINOVA_SCENE env var
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENES_DIR = os.path.join(_THIS_DIR, "scenes")


def discover_scenes(scenes_dir: str | None = None) -> list[str]:
    """Return sorted list of scene file paths in *scenes_dir*."""
    d = scenes_dir or DEFAULT_SCENES_DIR
    if not os.path.isdir(d):
        return []
    files = [
        os.path.join(d, f)
        for f in sorted(os.listdir(d))
        if f.endswith((".xml", ".mjcf"))
    ]
    return files


def select_scene(scenes_dir: str | None = None) -> str:
    """Pick a scene, either from ``KINOVA_SCENE`` env var or interactive menu.

    Returns:
        Absolute path to the selected scene file.
    """
    scenes = discover_scenes(scenes_dir)
    if not scenes:
        print(f"ERROR: No scene files found in {scenes_dir or DEFAULT_SCENES_DIR}")
        sys.exit(1)

    # ── Env-var shortcut ──────────────────────────────────────────────
    env = os.environ.get("KINOVA_SCENE", "").strip()
    if env:
        # Accept absolute path or bare filename
        if os.path.isabs(env) and os.path.isfile(env):
            return env
        # Try matching against discovered scenes by basename
        for s in scenes:
            if os.path.basename(s) == env:
                return s
        # Try as path relative to scenes dir
        candidate = os.path.join(scenes_dir or DEFAULT_SCENES_DIR, env)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        print(f"ERROR: KINOVA_SCENE='{env}' not found.")
        sys.exit(1)

    # ── Interactive menu ──────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════╗")
    print("║       Kinova Scene Selector              ║")
    print("╚══════════════════════════════════════════╝\n")
    print(f"  Discovered {len(scenes)} scene(s):\n")
    for i, path in enumerate(scenes, 1):
        name = os.path.basename(path)
        print(f"    [{i}]  {name}")
    print()

    while True:
        try:
            raw = input("  Select scene number: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(scenes):
                chosen = scenes[idx]
                print(f"\n  ✓ Selected: {os.path.basename(chosen)}\n")
                return chosen
            print(f"    ✗ Enter a number between 1 and {len(scenes)}.")
        except ValueError:
            print("    ✗ Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


if __name__ == "__main__":
    path = select_scene()
    print(f"Scene path: {path}")
