from __future__ import annotations

import threading
import unittest

from kinova_middleware.backend.interfaces.capabilities import BackendCapability
from kinova_middleware.backend.mcp_server.services import (
    GeometryToolService,
    KinovaMotionToolService,
    TaskPlanningToolService,
    ToolRuntimeContext,
)
from kinova_middleware.backend.mcp_server.tools import setup_tools


class FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class FakeController:
    def __init__(self) -> None:
        self.arm_dof = 4
        self.current_pos = [0.2, 0.0, 0.15]
        self.current_quat = [0.0, 0.0, 0.0, 1.0]
        self.pending_pos: list[float] | None = None
        self.sent_joint_targets: list[list[float]] = []
        self.gripper_commands: list[float] = []
        self.rotations: list[float] = []
        self.move_home_called = False
        self.reset_scene_called = False

    def move_home(self) -> None:
        self.move_home_called = True

    def reset_scene(self) -> None:
        self.reset_scene_called = True

    def get_end_effector_pose(self):
        return tuple(self.current_pos), tuple(self.current_quat)

    def solve_ik(self, target_pos, target_quat, seed_q_rad=None, move_wrist: bool = True):
        self.pending_pos = [float(v) for v in target_pos]
        self.current_quat = [float(v) for v in target_quat]
        return [0.1, 0.2, 0.3, 0.4]

    def solve_ik_position_only(self, target_pos, seed_q_rad=None, move_wrist: bool = True):
        self.pending_pos = [float(v) for v in target_pos]
        return [0.4, 0.3, 0.2, 0.1]

    def send_joint_position_rad(self, q_target) -> None:
        self.sent_joint_targets.append([float(v) for v in q_target])
        if self.pending_pos is not None:
            self.current_pos = self.pending_pos.copy()

    def get_target_joint_angles_rad(self):
        return [0.1, 0.2, 0.3, 0.4]

    def get_joint_angles_rad(self):
        return [0.1, 0.2, 0.3, 0.4]

    def get_joint_vel_rad(self):
        return [0.0, 0.0, 0.0, 0.0]

    def set_gripper_percent(self, percent: float) -> None:
        self.gripper_commands.append(float(percent))

    def get_gripper_state(self) -> dict:
        current = self.gripper_commands[-1] if self.gripper_commands else 0.2
        return {
            "percent": current,
            "target_percent": current,
            "max_pos_err": 0.0,
            "max_vel": 0.0,
            "settled": True,
        }

    def wait_for_gripper(self, **kwargs) -> bool:
        return True

    def get_finger_forces(self) -> dict:
        return {"forces": [0.1, 0.2], "max_abs_force": 0.2, "contact_detected": False}

    def rotate_wrist(self, angle_deg: float) -> None:
        self.rotations.append(float(angle_deg))

    def get_object_pose(self, body_name: str) -> dict:
        if body_name == "all":
            return {
                "status": "ok",
                "objects": [
                    {
                        "body_name": "cube",
                        "position": {"x": 0.1, "y": 0.0, "z": 0.12},
                        "size": [0.02, 0.02, 0.02],
                        "geom_type": "box",
                        "quaternion": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
                    }
                ],
            }
        return {
            "body_name": body_name,
            "position": {"x": 0.1, "y": 0.0, "z": 0.12},
            "size": [0.02, 0.02, 0.02],
            "geom_type": "box",
            "quaternion": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        }


class MCPToolRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = FakeController()
        self.mcp = FakeMCP()
        setup_tools(
            self.mcp,
            {
                "get_controller": lambda: self.controller,
                "motion_lock": threading.RLock(),
                "physics_lock": threading.RLock(),
                "run_until_reached": lambda **kwargs: True,
                "reset_or_reload_scene": lambda scene_number=None: {
                    "status": "ok",
                    "message": "Scene reset successfully.",
                    "scene_number": scene_number,
                },
            },
        )

    def test_expected_core_tools_are_registered(self) -> None:
        for tool_name in (
            "reset_scene",
            "move_home",
            "get_end_effector_pose",
            "set_gripper",
            "move_pose",
            "rotate_wrist",
            "get_object_pose",
            "plan_object_grasp",
            "plan_wrist_alignment",
            "plan_bin_place",
            "plan_stack_place",
        ):
            with self.subTest(tool=tool_name):
                self.assertIn(tool_name, self.mcp.tools)

    def test_move_home_uses_controller_and_reports_success(self) -> None:
        result = self.mcp.tools["move_home"]()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(self.controller.move_home_called)

    def test_get_end_effector_pose_returns_structured_pose(self) -> None:
        result = self.mcp.tools["get_end_effector_pose"]()

        self.assertEqual(result["position"]["x"], 0.2)
        self.assertEqual(result["position"]["z"], 0.15)
        self.assertEqual(result["quaternion"]["qw"], 1.0)

    def test_set_gripper_calls_controller(self) -> None:
        result = self.mcp.tools["set_gripper"](0.6)

        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(len(self.controller.gripper_commands), 1)
        self.assertAlmostEqual(self.controller.gripper_commands[-1], 0.6)

    def test_move_pose_uses_position_only_fallback_for_zero_quaternion(self) -> None:
        result = self.mcp.tools["move_pose"](
            target_pos=[0.2, 0.0, 0.14],
            target_quat=[0.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode_used"], "position_only")
        self.assertEqual(self.controller.sent_joint_targets[-1], [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(result["final_pose"]["position"], [0.2, 0.0, 0.14])

    def test_move_pose_rejects_out_of_workspace_target(self) -> None:
        result = self.mcp.tools["move_pose"](
            target_pos=[0.8, 0.0, 0.2],
            target_quat=[0.0, 0.0, 0.0, 1.0],
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("outside the workspace", result["message"])

    def test_rotate_wrist_calls_controller(self) -> None:
        result = self.mcp.tools["rotate_wrist"](45.0)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.controller.rotations, [45.0])

    def test_get_object_pose_uses_public_controller_method(self) -> None:
        result = self.mcp.tools["get_object_pose"]("cube")

        self.assertEqual(result["body_name"], "cube")
        self.assertEqual(result["geom_type"], "box")
        self.assertEqual(result["position"]["z"], 0.12)

    def test_optional_scene_and_object_tools_are_not_registered_without_capabilities(self) -> None:
        mcp = FakeMCP()
        registered = setup_tools(
            mcp,
            {
                "get_controller": lambda: self.controller,
                "motion_lock": threading.RLock(),
                "physics_lock": threading.RLock(),
                "run_until_reached": lambda **kwargs: True,
                "reset_or_reload_scene": lambda scene_number=None: {"status": "ok"},
                "capabilities": frozenset(
                    {
                        BackendCapability.ARM_MOTION,
                        BackendCapability.IK_SOLVER,
                        BackendCapability.GRIPPER_CONTROL,
                    }
                ),
            },
        )

        self.assertNotIn("reset_scene", mcp.tools)
        self.assertNotIn("get_object_pose", mcp.tools)
        self.assertNotIn("plan_object_grasp", mcp.tools)
        self.assertNotIn("plan_wrist_alignment", mcp.tools)
        self.assertNotIn("plan_bin_place", mcp.tools)
        self.assertNotIn("plan_stack_place", mcp.tools)
        self.assertIn("move_pose", mcp.tools)
        self.assertIn("compute_grasp_height", mcp.tools)
        self.assertEqual(
            registered,
            [
                "move_home",
                "get_end_effector_pose",
                "set_gripper",
                "move_pose",
                "rotate_wrist",
                "compute_grasp_height",
                "compute_wrist_alignment",
            ],
        )


class MCPApplicationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = FakeController()
        self.runtime = ToolRuntimeContext(
            get_controller=lambda: self.controller,
            motion_lock=threading.RLock(),
            physics_lock=threading.RLock(),
            run_until_reached=lambda **kwargs: True,
            reset_or_reload_scene=lambda scene_number=None: {
                "status": "ok",
                "message": "Scene reset successfully.",
                "scene_number": scene_number,
            },
        )
        self.motion_service = KinovaMotionToolService(self.runtime)
        self.geometry_service = GeometryToolService()
        self.task_planning_service = TaskPlanningToolService(self.runtime, self.geometry_service)

    def test_motion_service_move_pose_keeps_existing_behavior(self) -> None:
        result = self.motion_service.move_pose(
            target_pos=[0.2, 0.0, 0.14],
            target_quat=[0.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode_used"], "position_only")
        self.assertEqual(self.controller.sent_joint_targets[-1], [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(result["final_pose"]["position"], [0.2, 0.0, 0.14])

    def test_motion_service_reset_scene_uses_runtime_hook(self) -> None:
        result = self.motion_service.reset_scene(2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scene_number"], 2)

    def test_geometry_service_parses_string_inputs(self) -> None:
        result = self.geometry_service.compute_grasp_height(
            "box",
            "[0.02, 0.02, 0.02]",
            "[0.0, 0.0, 0.0, 1.0]",
        )

        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["top_height"], 0.0)

    def test_task_planning_service_plans_grasp_profile(self) -> None:
        result = self.task_planning_service.plan_object_grasp("red_cube", profile="sort_cubes")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recommended_gripper_percent"], 0.54)
        self.assertEqual(result["approach_move"]["target_pos"], [0.1, 0.0, 0.16])
        self.assertEqual(result["grasp_move"]["target_pos"], [0.1, 0.0, 0.075])
        self.assertFalse(result["lift_move"]["move_wrist"])

    def test_task_planning_service_plans_box_wrist_alignment(self) -> None:
        result = self.task_planning_service.plan_wrist_alignment(
            "cube",
            [0.0, 0.0, 0.0, 1.0],
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["snapped_for_box"])
        self.assertIn("angle_deg", result)

    def test_task_planning_service_inferrs_bin_target(self) -> None:
        result = self.task_planning_service.plan_bin_place("red_cube")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["target_name"], "red_bin_target")
        self.assertEqual(result["place_move"]["target_pos"], [0.1, 0.0, 0.12])

    def test_task_planning_service_plans_stack_place(self) -> None:
        result = self.task_planning_service.plan_stack_place("red_cube", "blue_cube")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stack_center_z"], 0.17)
        self.assertEqual(result["preplace_move"]["target_pos"], [0.1, 0.0, 0.27])
        self.assertEqual(result["place_move"]["target_pos"], [0.1, 0.0, 0.21])
        self.assertFalse(result["retreat_move"]["move_wrist"])
