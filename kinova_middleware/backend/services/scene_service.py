from __future__ import annotations

from typing import Callable

import numpy as np

from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter


class MuJoCoSceneService:
    """Own scene reset and end-effector pose queries for the MuJoCo runtime."""

    def __init__(
        self,
        runtime: MuJoCoRuntimeAdapter,
        *,
        ee_pose_provider: Callable[[object], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self._runtime = runtime
        self._ee_pose_provider = ee_pose_provider

    def reset(self, initial_keyframe: str | None = None) -> None:
        self._runtime.reset(initial_keyframe=initial_keyframe)

    def get_end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        pos, quat_wxyz = self._ee_pose_provider(self._runtime.data)
        quat_xyzw = (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )
        pos_xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
        return (pos_xyz, quat_xyzw)
