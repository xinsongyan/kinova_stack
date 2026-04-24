from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GripperControl(Protocol):
    def set_gripper_percent(self, percent: float) -> None:
        ...

    def get_gripper_state(self) -> dict:
        ...


Gripper = GripperControl
