from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SceneControl(Protocol):
    def reset_scene(self) -> None:
        ...


Scene = SceneControl
