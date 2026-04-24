from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class ArmMotion(Protocol):
    def move_home(self) -> None:
        ...

    def send_joint_position_rad(self, q_des: Sequence[float]) -> None:
        ...

    def get_joint_angles_rad(self) -> list[float]:
        ...

    def get_joint_vel_rad(self) -> list[float]:
        ...


Arm = ArmMotion
