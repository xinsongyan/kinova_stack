from __future__ import annotations

import importlib
import unittest


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


class ImportSmokeTests(unittest.TestCase):
    def test_core_package_imports(self) -> None:
        modules = [
            "kinova_middleware",
            "kinova_middleware.backend.kinova_backend",
            "kinova_middleware.backend.kinova_controller",
            "kinova_middleware.backend.controller",
            "kinova_middleware.backend.factory",
            "kinova_middleware.backend.interfaces",
            "kinova_middleware.backend.interfaces.arm",
            "kinova_middleware.backend.interfaces.capabilities",
            "kinova_middleware.backend.interfaces.gripper",
            "kinova_middleware.backend.interfaces.ik",
            "kinova_middleware.backend.interfaces.object_query",
            "kinova_middleware.backend.interfaces.protocols",
            "kinova_middleware.backend.interfaces.scene",
            "kinova_middleware.backend.config",
            "kinova_middleware.backend.config.robot_model",
            "kinova_middleware.backend.config.kinova_gen3_lite",
            "kinova_middleware.backend.adapters",
            "kinova_middleware.backend.mujoco_config",
            "kinova_middleware.backend.mcp_server.services",
            "kinova_middleware.backend.mcp_server.tool_registry",
            "kinova_middleware.backend.mcp_server.tools",
            "kinova_middleware.backend.mcp_server.prompts",
            "kinova_middleware.backend.mcp_server.app",
            "kinova_middleware.backend.mcp_server.toolsets",
            "kinova_middleware.backend.mcp_server.toolsets.motion_tools",
            "kinova_middleware.backend.mcp_server.toolsets.scene_tools",
            "kinova_middleware.backend.mcp_server.toolsets.task_prompts",
            "kinova_middleware.llm_clients.rate_limits",
            "kinova_middleware.llm_clients.scene_tools",
            "kinova_middleware.llm_clients.status_reporting",
            "kinova_middleware.llm_clients.tool_defs",
            "kinova_middleware.llm_clients.tool_dispatch",
            "kinova_middleware.llm_clients.tool_schema",
            "kinova_middleware.llm_clients.workflow_runner",
            "kinova_middleware.llm_clients.ultimate_llm",
            "kinova_middleware.scenes.scene_selector",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

    def test_sim_support_modules_import_when_mujoco_is_available(self) -> None:
        if not _module_available("mujoco"):
            self.skipTest("MuJoCo is not installed in this Python environment.")

        modules = [
            "kinova_sim.controller",
            "kinova_sim.governor",
            "kinova_middleware.backend.runtime",
            "kinova_middleware.backend.runtime.mujoco_runtime",
            "kinova_middleware.backend.services",
            "kinova_middleware.backend.services.motion_service",
            "kinova_middleware.backend.services.gripper_service",
            "kinova_middleware.backend.services.scene_service",
            "kinova_middleware.backend.services.object_query_service",
            "kinova_middleware.backend.adapters.mujoco_arm_adapter",
            "kinova_middleware.backend.kinova_mujoco_backend",
            "kinova_middleware.backend.mujoco_control",
            "kinova_middleware.backend.mujoco_ik",
            "kinova_middleware.backend.mujoco_runtime",
            "kinova_sim.trajectory",
            "kinova_sim.sim_env",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)
