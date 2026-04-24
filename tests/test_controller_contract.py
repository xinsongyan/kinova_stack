from __future__ import annotations

import unittest

from kinova_middleware.backend.kinova_backend import CartesianPose, KinovaBackend, SafetyWrapperBackend
from kinova_middleware.backend.controller import KinovaController


class FakeBackend(KinovaBackend):
    def __init__(self) -> None:
        self.events: list[tuple[str, object | None]] = []
        self._joints = [0.0, 0.0, 0.0, 0.0]
        self._target = [0.0, 0.0, 0.0, 0.0]
        self._quat = (0.0, 0.0, 0.0, 1.0)
        self._pos = (0.2, 0.0, 0.15)

    @property
    def dof(self) -> int:
        return 4

    @property
    def arm_dof(self) -> int:
        return 4

    def init(self) -> None:
        self.events.append(("init", None))

    def close(self) -> None:
        self.events.append(("close", None))

    def move_home(self) -> None:
        self.events.append(("move_home", None))
        self._target = [0.0, 0.0, 0.0, 0.0]

    def send_joint_position_rad(self, q_des):
        q_list = [float(v) for v in q_des]
        self.events.append(("send_joint_position_rad", q_list))
        self._target = q_list
        self._joints = q_list.copy()

    def get_end_effector_pose(self):
        return self._pos, self._quat

    def solve_ik(self, target_pos, target_quat, q_seed=None, move_wrist: bool = True):
        self.events.append(
            (
                "solve_ik",
                {
                    "target_pos": list(target_pos),
                    "target_quat": list(target_quat),
                    "q_seed": None if q_seed is None else list(q_seed),
                    "move_wrist": move_wrist,
                },
            )
        )
        return [0.1, 0.2, 0.3, 0.4]

    def step(self, **kwargs):
        self.events.append(("step", kwargs))
        return True

    def is_reached(self, **kwargs):
        self.events.append(("is_reached", kwargs))
        return True

    def reset_scene(self) -> None:
        self.events.append(("reset_scene", None))

    def get_joint_angles_rad(self):
        return self._joints.copy()

    def get_target_joint_angles_rad(self):
        return self._target.copy()

    def get_joint_vel_rad(self):
        return [0.0, 0.0, 0.0, 0.0]

    def set_gripper_percent(self, percent: float) -> None:
        self.events.append(("set_gripper_percent", float(percent)))

    def get_object_pose(self, body_name: str) -> dict:
        result = {
            "body_name": body_name,
            "position": {"x": 0.1, "y": 0.0, "z": 0.2},
            "size": [0.02],
            "geom_type": "box",
            "quaternion": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        }
        self.events.append(("get_object_pose", body_name))
        return result


class KinovaControllerContractTests(unittest.TestCase):
    def test_controller_wraps_plain_backend_with_safety_wrapper(self) -> None:
        backend = FakeBackend()
        controller = KinovaController(backend)

        self.assertIsInstance(controller._backend, SafetyWrapperBackend)

    def test_controller_can_preserve_existing_backend_without_wrapping(self) -> None:
        backend = FakeBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        self.assertIs(controller._backend, backend)

    def test_controller_delegates_motion_and_gripper_calls(self) -> None:
        backend = FakeBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        controller.init()
        controller.move_home()
        controller.send_joint_position_rad([0.1, 0.2, 0.3, 0.4])
        controller.set_gripper_percent(0.6)
        controller.reset_scene()
        controller.close()

        self.assertIn(("init", None), backend.events)
        self.assertIn(("move_home", None), backend.events)
        self.assertIn(("send_joint_position_rad", [0.1, 0.2, 0.3, 0.4]), backend.events)
        self.assertIn(("set_gripper_percent", 0.6), backend.events)
        self.assertIn(("reset_scene", None), backend.events)
        self.assertIn(("close", None), backend.events)

    def test_controller_supports_polymorphic_backend_for_pose_planning(self) -> None:
        backend = FakeBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)
        pose = CartesianPose(0.25, 0.05, 0.12, 0.0, 0.0, 0.0, 1.0)

        q_target = controller.move_to_pose(pose)

        self.assertEqual(q_target, [0.1, 0.2, 0.3, 0.4])
        self.assertIn(
            (
                "solve_ik",
                {
                    "target_pos": [0.25, 0.05, 0.12],
                    "target_quat": [0.0, 0.0, 0.0, 1.0],
                    "q_seed": None,
                    "move_wrist": True,
                },
            ),
            backend.events,
        )
        self.assertIn(("send_joint_position_rad", [0.1, 0.2, 0.3, 0.4]), backend.events)

    def test_controller_delegates_scene_object_queries(self) -> None:
        backend = FakeBackend()
        controller = KinovaController(backend, enforce_safety_wrapper=False)

        result = controller.get_object_pose("cube")

        self.assertEqual(result["body_name"], "cube")
        self.assertIn(("get_object_pose", "cube"), backend.events)
