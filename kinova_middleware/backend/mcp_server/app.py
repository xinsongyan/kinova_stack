from __future__ import annotations

import os
import sys


if __package__ in (None, ""):
    # Preserve direct-script execution from the repo root while using package imports.
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    from kinova_middleware.backend.mcp_server.server import main as server_main

    server_main()


def __getattr__(name: str):
    if name == "mcp":
        from kinova_middleware.backend.mcp_server.server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "mcp"]


if __name__ == "__main__":
    main()
