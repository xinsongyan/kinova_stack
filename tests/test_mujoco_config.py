from __future__ import annotations

import unittest

from kinova_middleware.backend.config.kinova_gen3_lite import (
    DEFAULT_EE_BODY,
    DEFAULT_FINGER_JOINTS,
    DEFAULT_JOINT_ORDER,
    DEFAULT_KINOVA_MUJOCO_CONFIG,
)
from kinova_middleware.backend.config.robot_model import (
    MuJoCoRobotConfig,
)


class MuJoCoConfigTests(unittest.TestCase):
    def test_default_config_matches_legacy_defaults(self) -> None:
        config = DEFAULT_KINOVA_MUJOCO_CONFIG

        self.assertIsInstance(config, MuJoCoRobotConfig)
        self.assertEqual(config.joint_names, DEFAULT_JOINT_ORDER)
        self.assertEqual(config.finger_joint_names, DEFAULT_FINGER_JOINTS)
        self.assertEqual(config.ee_body, DEFAULT_EE_BODY)
        self.assertEqual(config.initial_keyframe, "home")
        self.assertEqual(config.v_max_arm, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(config.kp_arm[0], 500.0)

    def test_config_with_overrides_returns_new_config(self) -> None:
        updated = DEFAULT_KINOVA_MUJOCO_CONFIG.with_overrides(
            model_path="custom_scene.xml",
            joint_names=("j1", "j2", "f1"),
            finger_joint_names=("f1",),
            ee_body="tool",
            ee_site="tool_site",
            initial_keyframe="ready",
        )

        self.assertEqual(updated.model_path, "custom_scene.xml")
        self.assertEqual(updated.joint_names, ("j1", "j2", "f1"))
        self.assertEqual(updated.finger_joint_names, ("f1",))
        self.assertEqual(updated.ee_body, "tool")
        self.assertEqual(updated.ee_site, "tool_site")
        self.assertEqual(updated.initial_keyframe, "ready")
        self.assertEqual(DEFAULT_KINOVA_MUJOCO_CONFIG.ee_site, None)

    def test_config_with_overrides_can_explicitly_disable_optional_fields(self) -> None:
        seeded = DEFAULT_KINOVA_MUJOCO_CONFIG.with_overrides(
            ee_site="tool_site",
            initial_keyframe="ready",
        )

        disabled = seeded.with_overrides(ee_site=None, initial_keyframe=None)

        self.assertIsNone(disabled.ee_site)
        self.assertIsNone(disabled.initial_keyframe)

    def test_config_with_none_keeps_non_optional_defaults(self) -> None:
        updated = DEFAULT_KINOVA_MUJOCO_CONFIG.with_overrides(
            model_path=None,
            joint_names=None,
            finger_joint_names=None,
            ee_body=None,
        )

        self.assertEqual(updated.model_path, DEFAULT_KINOVA_MUJOCO_CONFIG.model_path)
        self.assertEqual(updated.joint_names, DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names)
        self.assertEqual(
            updated.finger_joint_names,
            DEFAULT_KINOVA_MUJOCO_CONFIG.finger_joint_names,
        )
        self.assertEqual(updated.ee_body, DEFAULT_KINOVA_MUJOCO_CONFIG.ee_body)
