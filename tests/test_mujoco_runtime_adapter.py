from __future__ import annotations

import importlib
import unittest


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


class MuJoCoRuntimeAdapterTests(unittest.TestCase):
    def test_runtime_opens_default_model_without_viewer(self) -> None:
        if not _module_available("mujoco"):
            self.skipTest("MuJoCo is not installed in this Python environment.")

        from kinova_middleware.backend.mujoco_config import DEFAULT_KINOVA_MUJOCO_CONFIG
        from kinova_middleware.backend.mujoco_runtime import MuJoCoRuntimeAdapter

        runtime = MuJoCoRuntimeAdapter(
            model_path=DEFAULT_KINOVA_MUJOCO_CONFIG.model_path,
            joint_names=DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names,
            ee_body_name=DEFAULT_KINOVA_MUJOCO_CONFIG.ee_body,
            ee_site_name=DEFAULT_KINOVA_MUJOCO_CONFIG.ee_site,
            viewer=False,
            site_candidates=("tool_tip", "ee_site", "end_effector"),
        )

        runtime.open(initial_keyframe=DEFAULT_KINOVA_MUJOCO_CONFIG.initial_keyframe)
        try:
            self.assertEqual(len(runtime.joint_ids), len(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names))
            self.assertEqual(len(runtime.qpos_adr), len(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names))
            self.assertEqual(len(runtime.actuator_ids), len(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names))
            self.assertIsNotNone(runtime.model)
            self.assertIsNotNone(runtime.data)
            self.assertGreater(runtime.ee_body_id, -1)
        finally:
            runtime.close()
