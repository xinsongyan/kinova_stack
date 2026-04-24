from __future__ import annotations

import importlib
import unittest


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


class MuJoCoIKServiceTests(unittest.TestCase):
    def test_service_solves_current_pose_to_finite_arm_configuration(self) -> None:
        if not _module_available("mujoco"):
            self.skipTest("MuJoCo is not installed in this Python environment.")

        import numpy as np

        from kinova_middleware.backend.config.kinova_gen3_lite import DEFAULT_KINOVA_MUJOCO_CONFIG
        from kinova_middleware.backend.mujoco_ik import MuJoCoIKService
        from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter

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
            finger_names = set(DEFAULT_KINOVA_MUJOCO_CONFIG.finger_joint_names)
            arm_indices = [
                idx
                for idx, name in enumerate(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names)
                if name not in finger_names
            ]
            service = MuJoCoIKService(runtime, arm_indices)
            site_id = runtime.ee_site_id
            self.assertIsNotNone(site_id)

            target_pos = np.array(runtime.data.site_xpos[site_id], dtype=float)
            q_solution = service.solve(target_pos, None, None, active_dof=len(arm_indices))

            self.assertEqual(len(q_solution), len(arm_indices))
            self.assertTrue(np.all(np.isfinite(q_solution)))
            self.assertIsNotNone(service.last_iterations)
        finally:
            runtime.close()
