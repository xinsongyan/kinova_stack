from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectQuery(Protocol):
    def get_object_pose(self, name: str) -> dict:
        ...
