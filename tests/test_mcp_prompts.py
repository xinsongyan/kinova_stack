from __future__ import annotations

import unittest

from kinova_middleware.backend.interfaces.capabilities import BackendCapability
from kinova_middleware.backend.mcp_server.prompts import setup_prompts


class FakeMCP:
    def __init__(self) -> None:
        self.prompts: dict[str, object] = {}

    @property
    def prompt(self):
        def decorator(fn):
            self.prompts[fn.__name__] = fn
            return fn

        return decorator


class MCPPromptRegistrationTests(unittest.TestCase):
    def test_prompts_register_and_return_expected_content(self) -> None:
        mcp = FakeMCP()
        setup_prompts(mcp)

        self.assertIn("grab_shapes", mcp.prompts)
        self.assertIn("sort_cubes", mcp.prompts)
        self.assertIn("stack_cubes", mcp.prompts)

        grab_prompt = mcp.prompts["grab_shapes"]()
        stack_prompt = mcp.prompts["stack_cubes"]("red_cube", "green_cube")
        sort_prompt = mcp.prompts["sort_cubes"]()

        self.assertIn("move_home()", grab_prompt)
        self.assertIn("plan_object_grasp", grab_prompt)
        self.assertIn("plan_wrist_alignment", grab_prompt)
        self.assertIn("plan_bin_place", sort_prompt)
        self.assertIn("plan_stack_place", stack_prompt)
        self.assertIn("green_cube", stack_prompt)
        self.assertIn("red_cube", stack_prompt)

    def test_prompts_are_withheld_without_object_query_capability(self) -> None:
        mcp = FakeMCP()
        registered = setup_prompts(
            mcp,
            capabilities=frozenset(
                {
                    BackendCapability.ARM_MOTION,
                    BackendCapability.IK_SOLVER,
                    BackendCapability.GRIPPER_CONTROL,
                }
            ),
        )

        self.assertEqual(registered, [])
        self.assertEqual(mcp.prompts, {})
