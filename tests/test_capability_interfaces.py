from __future__ import annotations

import unittest

from kinova_middleware.backend.interfaces import (
    ArmMotion,
    BackendCapability,
    GripperControl,
    IKSolver,
    ObjectQuery,
    SceneControl,
    get_supported_capabilities,
    supports_capability,
)
from kinova_middleware.backend.kinova_backend import KinovaBackend, SafetyWrapperBackend
from kinova_middleware.backend.kinova_controller import KinovaController


class MinimalBackend(KinovaBackend):
    @property
    def dof(self) -> int:
        return 4

    @property
    def arm_dof(self) -> int:
        return 4

    def init(self) -> None:
        pass

    def close(self) -> None:
        pass

    def move_home(self) -> None:
        pass

    def send_joint_position_rad(self, q_des) -> None:
        pass

    def get_end_effector_pose(self):
        return (0.2, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)

    def solve_ik(self, target_pos, target_quat, q_seed=None, move_wrist: bool = True):
        return [0.1, 0.2, 0.3, 0.4]

    def step(self, **kwargs):
        return True

    def is_reached(self, **kwargs):
        return True

    def reset_scene(self) -> None:
        pass

    def get_joint_angles_rad(self):
        return [0.0, 0.0, 0.0, 0.0]

    def get_target_joint_angles_rad(self):
        return [0.0, 0.0, 0.0, 0.0]

    def get_joint_vel_rad(self):
        return [0.0, 0.0, 0.0, 0.0]

    def set_gripper_percent(self, percent: float) -> None:
        pass


class ObjectQueryBackend(MinimalBackend):
    def get_object_pose(self, body_name: str) -> dict:
        return {
            "body_name": body_name,
            "position": {"x": 0.1, "y": 0.0, "z": 0.2},
            "size": [0.02, 0.02, 0.02],
            "geom_type": "box",
            "quaternion": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        }


class CapabilityInterfaceTests(unittest.TestCase):
    def test_core_protocols_apply_to_backend_and_controller(self) -> None:
        backend = MinimalBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        for obj in (backend, controller):
            with self.subTest(obj=type(obj).__name__, protocol="ArmMotion"):
                self.assertIsInstance(obj, ArmMotion)
            with self.subTest(obj=type(obj).__name__, protocol="IKSolver"):
                self.assertIsInstance(obj, IKSolver)
            with self.subTest(obj=type(obj).__name__, protocol="GripperControl"):
                self.assertIsInstance(obj, GripperControl)
            with self.subTest(obj=type(obj).__name__, protocol="SceneControl"):
                self.assertIsInstance(obj, SceneControl)

    def test_object_query_protocol_exists_but_capability_is_not_advertised_by_default(self) -> None:
        backend = MinimalBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        # Structural typing alone is not enough for optional features because
        # the legacy backend facade still carries a default get_object_pose.
        self.assertIsInstance(backend, ObjectQuery)
        self.assertFalse(supports_capability(backend, BackendCapability.OBJECT_QUERY))
        self.assertFalse(supports_capability(controller, "object_query"))

    def test_object_query_capability_flows_through_wrapper_and_controller(self) -> None:
        backend = ObjectQueryBackend()
        wrapped = SafetyWrapperBackend(backend)
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        expected = frozenset(
            {
                BackendCapability.ARM_MOTION,
                BackendCapability.IK_SOLVER,
                BackendCapability.GRIPPER_CONTROL,
                BackendCapability.SCENE_CONTROL,
                BackendCapability.OBJECT_QUERY,
            }
        )

        self.assertEqual(get_supported_capabilities(backend), expected)
        self.assertEqual(get_supported_capabilities(wrapped), expected)
        self.assertEqual(get_supported_capabilities(controller), expected)
        self.assertTrue(supports_capability(wrapped, BackendCapability.OBJECT_QUERY))
        self.assertTrue(controller.supports_capability("object_query"))
