from __future__ import annotations

from types import SimpleNamespace
import unittest

from langchain_core.messages import AIMessage

from kinova_middleware.llm_clients.local_tools.tool_defs import (
    CHECK_SORTING_STATUS_TOOL,
    FINISH_TASK_TOOL,
)
from kinova_middleware.llm_clients.runtime.workflow_runner import (
    build_reasoned_agent_session,
    run_reasoned_agent_loop,
)


class FakeLLMWithTools:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages):
        if not self._responses:
            raise AssertionError("No more fake LLM responses available.")
        return self._responses.pop(0)


class FakeLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self.bound_tools = None
        self.bind_kwargs = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bind_kwargs = kwargs
        return FakeLLMWithTools(self._responses)


class FakeMCPClient:
    async def list_tools(self):
        return []

    async def list_prompts(self):
        return []


class ReasonedWorkflowRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_reasoned_agent_session_includes_local_tools(self) -> None:
        llm = FakeLLM([])
        client = FakeMCPClient()

        session = await build_reasoned_agent_session(
            client,
            llm,
            system_prompt="system prompt",
            task="do the task",
            extra_tools=[FINISH_TASK_TOOL],
            tool_choice="required",
        )

        self.assertEqual(len(session.messages), 3)
        self.assertIn("finish_task", session.action_reference)
        self.assertIsNotNone(llm.bound_tools)
        self.assertEqual(llm.bind_kwargs["tool_choice"], "required")

    async def test_run_reasoned_agent_loop_handles_finish_task(self) -> None:
        llm = FakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "call_tool_with_reason",
                            "args": {
                                "reason": "I am stopping because the task is done.",
                                "tool_name": "finish_task",
                                "tool_args": {"summary": "done"},
                            },
                        }
                    ],
                )
            ]
        )
        client = FakeMCPClient()
        session = await build_reasoned_agent_session(
            client,
            llm,
            system_prompt="system prompt",
            task="do the task",
            extra_tools=[FINISH_TASK_TOOL],
            tool_choice="required",
        )

        result = await run_reasoned_agent_loop(
            client,
            session.llm_with_tools,
            session.messages,
            max_steps=2,
        )

        self.assertEqual(result.stop_reason, "finish_task")
        self.assertEqual(result.model_summary, "done")

    async def test_run_reasoned_agent_loop_dispatches_local_tool_handlers(self) -> None:
        llm = FakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "call_tool_with_reason",
                            "args": {
                                "reason": "I am checking progress first.",
                                "tool_name": "check_sorting_status",
                                "tool_args": {},
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-2",
                            "name": "call_tool_with_reason",
                            "args": {
                                "reason": "I am stopping because the task is complete.",
                                "tool_name": "finish_task",
                                "tool_args": {"summary": "sorted"},
                            },
                        }
                    ],
                ),
            ]
        )
        client = FakeMCPClient()
        session = await build_reasoned_agent_session(
            client,
            llm,
            system_prompt="system prompt",
            task="do the task",
            extra_tools=[CHECK_SORTING_STATUS_TOOL, FINISH_TASK_TOOL],
            tool_choice="required",
        )
        handler_calls: list[str] = []

        async def fake_check_sorting_status(_client, _tool_args):
            handler_calls.append("called")
            return "CURRENT SORTING STATUS REPORT:\n- red_cube: At Starting Position"

        result = await run_reasoned_agent_loop(
            client,
            session.llm_with_tools,
            session.messages,
            max_steps=3,
            local_tool_handlers={"check_sorting_status": fake_check_sorting_status},
        )

        self.assertEqual(handler_calls, ["called"])
        self.assertEqual(result.stop_reason, "finish_task")
        self.assertEqual(result.model_summary, "sorted")
