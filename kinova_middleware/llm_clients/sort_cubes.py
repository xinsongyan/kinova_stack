#!/usr/bin/env python3
"""Compatibility entrypoint for the sort_cubes agent."""

from __future__ import annotations

import asyncio
import os
import sys

if __package__ in (None, ""):
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from kinova_middleware.llm_clients.agents.sort_cubes import main


if __name__ == "__main__":
    asyncio.run(main())
