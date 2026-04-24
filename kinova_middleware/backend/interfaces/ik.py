from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class IKSolver(Protocol):
    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        ...

    def solve_ik_position_only(
        self,
        target_pos: Sequence[float],
        q_seed: Sequence[float] | None = None,
        move_wrist: bool = True,
    ) -> list[float]:
        ...


IK = IKSolver
