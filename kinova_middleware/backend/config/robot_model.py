from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


_UNSET = object()


@dataclass(frozen=True, slots=True)
class RobotModelConfig:
    """Robot/model-specific MuJoCo backend configuration."""

    model_path: str
    joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    ee_body: str
    ee_site: str | None
    initial_keyframe: str | None
    v_max_arm: tuple[float, ...]
    a_max_arm: tuple[float, ...]
    j_max_arm: tuple[float, ...]
    omega_arm: tuple[float, ...]
    kp_arm: tuple[float, ...]
    kd_arm: tuple[float, ...]
    kp_finger: float
    kd_finger: float
    finger_bias_mode: str
    control_dt: float
    j1_torque_limit: float

    def with_overrides(
        self,
        *,
        model_path: str | object = _UNSET,
        joint_names: Sequence[str] | object = _UNSET,
        finger_joint_names: Sequence[str] | object = _UNSET,
        ee_body: str | object = _UNSET,
        ee_site: str | None | object = _UNSET,
        initial_keyframe: str | None | object = _UNSET,
    ) -> "RobotModelConfig":
        return replace(
            self,
            model_path=(
                self.model_path if model_path is _UNSET or model_path is None else str(model_path)
            ),
            joint_names=(
                self.joint_names if joint_names is _UNSET or joint_names is None else tuple(joint_names)
            ),
            finger_joint_names=(
                self.finger_joint_names
                if finger_joint_names is _UNSET or finger_joint_names is None
                else tuple(finger_joint_names)
            ),
            ee_body=self.ee_body if ee_body is _UNSET or ee_body is None else str(ee_body),
            ee_site=self.ee_site if ee_site is _UNSET else ee_site,
            initial_keyframe=(
                self.initial_keyframe if initial_keyframe is _UNSET else initial_keyframe
            ),
        )


MuJoCoRobotConfig = RobotModelConfig
