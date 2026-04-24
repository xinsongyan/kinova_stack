from __future__ import annotations

import importlib
import unittest


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


class MuJoCoControlServiceTests(unittest.TestCase):
    def test_arm_and_gripper_services_produce_finite_commands(self) -> None:
        if not _module_available("mujoco"):
            self.skipTest("MuJoCo is not installed in this Python environment.")

        import numpy as np

        from kinova_middleware.backend.mujoco_config import DEFAULT_KINOVA_MUJOCO_CONFIG
        from kinova_middleware.backend.mujoco_control import (
            MuJoCoArmControlService,
            MuJoCoGripperService,
        )
        from kinova_middleware.backend.mujoco_ik import wrap_to_pi
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
            finger_names = set(DEFAULT_KINOVA_MUJOCO_CONFIG.finger_joint_names)
            arm_indices = [
                idx
                for idx, name in enumerate(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names)
                if name not in finger_names
            ]
            finger_indices = [
                idx
                for idx, name in enumerate(DEFAULT_KINOVA_MUJOCO_CONFIG.joint_names)
                if name in finger_names
            ]
            q_current = np.array([float(runtime.data.qpos[adr]) for adr in runtime.qpos_adr], dtype=float)

            arm_service = MuJoCoArmControlService(
                runtime,
                arm_indices,
                wrap_angle_fn=wrap_to_pi,
                v_max_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.v_max_arm,
                a_max_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.a_max_arm,
                j_max_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.j_max_arm,
                omega_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.omega_arm,
                kp_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.kp_arm,
                kd_arm=DEFAULT_KINOVA_MUJOCO_CONFIG.kd_arm,
                control_dt=DEFAULT_KINOVA_MUJOCO_CONFIG.control_dt,
                j1_torque_limit=DEFAULT_KINOVA_MUJOCO_CONFIG.j1_torque_limit,
            )
            gripper_service = MuJoCoGripperService(
                runtime,
                finger_indices,
                kp_finger=DEFAULT_KINOVA_MUJOCO_CONFIG.kp_finger,
                kd_finger=DEFAULT_KINOVA_MUJOCO_CONFIG.kd_finger,
                finger_bias_mode=DEFAULT_KINOVA_MUJOCO_CONFIG.finger_bias_mode,
            )

            q_target = arm_service.set_joint_target(
                [q_current[i] for i in arm_indices],
                q_current,
                q_current,
            )
            arm_commands = arm_service.compute_actuator_commands()
            self.assertEqual(len(arm_commands), len(arm_indices))
            self.assertTrue(all(np.isfinite(command) for _, command in arm_commands))
            self.assertTrue(arm_service.is_reached(q_target, pos_tol_rad=1.0, vel_tol_rad_s=10.0))

            q_target = gripper_service.set_target_percent(0.5, q_target)
            finger_commands = gripper_service.compute_actuator_commands(q_target)
            self.assertEqual(len(finger_commands), len(finger_indices))
            self.assertTrue(all(np.isfinite(command) for _, command in finger_commands))

            state = gripper_service.get_gripper_state(q_target)
            self.assertIn("percent", state)
            self.assertIn("settled", state)

            forces = gripper_service.get_finger_forces()
            self.assertEqual(len(forces["forces"]), len(finger_indices))
        finally:
            runtime.close()
