"""Tests for AgentTool — sub-agent spawning."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.events import AgentDone, AgentError
from mini_agent.tools.agent_tool import AgentTool
from mini_agent.tools.base import Tool, ToolResult


# ── Helpers ──────────────────────────────────────────────────────────────────


class DummyTool(Tool):
    """A no-op tool for testing tool subset filtering."""

    def __init__(self, tool_name: str):
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy {self._name}"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content="ok")


def _make_tools(*names: str) -> dict[str, Tool]:
    return {n: DummyTool(n) for n in names}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAgentToolSchema:
    """Test schema generation."""

    def test_name(self):
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash", "read"),
        )
        assert tool.name == "agent"

    def test_parameters_list_available_tools(self):
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash", "read", "write"),
        )
        schema = tool.parameters
        enum_values = schema["properties"]["tools"]["items"]["enum"]
        assert sorted(enum_values) == ["bash", "read", "write"]

    def test_to_schema_round_trip(self):
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash"),
        )
        s = tool.to_schema()
        assert s["name"] == "agent"
        assert "input_schema" in s


class TestAgentToolExecution:
    """Test sub-agent spawning and result handling."""

    @pytest.mark.asyncio
    async def test_unknown_tools_rejected(self):
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash", "read"),
        )
        result = await tool.execute(prompt="do stuff", tools=["nonexistent"])
        assert not result.success
        assert "Unknown tools" in result.error

    @pytest.mark.asyncio
    async def test_successful_sub_agent(self):
        """Sub-agent completes and returns content."""
        mock_llm = MagicMock()

        tool = AgentTool(
            llm_client=mock_llm,
            available_tools=_make_tools("bash", "read"),
        )

        # Patch Agent to avoid real LLM calls
        async def fake_run_stream(*args, **kwargs):
            yield AgentDone(content="Task completed successfully", steps=2)

        with patch("mini_agent.agent.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_stream = fake_run_stream
            instance.add_user_message = MagicMock()

            result = await tool.execute(prompt="find all .py files")

        assert result.success
        assert result.content == "Task completed successfully"
        # Verify sub-agent was created with the shared llm_client
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs
        assert call_kwargs["llm_client"] is mock_llm

    @pytest.mark.asyncio
    async def test_sub_agent_error(self):
        """Sub-agent encounters an error."""
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash"),
        )

        async def fake_run_stream(*args, **kwargs):
            yield AgentError(error="LLM call failed: timeout", steps=1)

        with patch("mini_agent.agent.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_stream = fake_run_stream
            instance.add_user_message = MagicMock()

            result = await tool.execute(prompt="do stuff")

        assert not result.success
        assert "timeout" in result.error

    @pytest.mark.asyncio
    async def test_sub_agent_exception(self):
        """Sub-agent raises an unexpected exception."""
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash"),
        )

        async def fake_run_stream(*args, **kwargs):
            raise RuntimeError("connection lost")
            yield  # make it a generator

        with patch("mini_agent.agent.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_stream = fake_run_stream
            instance.add_user_message = MagicMock()

            result = await tool.execute(prompt="do stuff")

        assert not result.success
        assert "connection lost" in result.error

    @pytest.mark.asyncio
    async def test_default_excludes_agent_tool(self):
        """When no tools specified, 'agent' is excluded to prevent recursion."""
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash", "read", "agent"),
        )

        async def fake_run_stream(*args, **kwargs):
            yield AgentDone(content="done", steps=1)

        with patch("mini_agent.agent.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_stream = fake_run_stream
            instance.add_user_message = MagicMock()

            await tool.execute(prompt="test")

        call_kwargs = MockAgent.call_args.kwargs
        sub_tool_names = [t.name for t in call_kwargs["tools"]]
        assert "agent" not in sub_tool_names
        assert "bash" in sub_tool_names
        assert "read" in sub_tool_names

    @pytest.mark.asyncio
    async def test_explicit_tool_subset(self):
        """When tools are specified explicitly, only those are passed."""
        tool = AgentTool(
            llm_client=MagicMock(),
            available_tools=_make_tools("bash", "read", "write"),
        )

        async def fake_run_stream(*args, **kwargs):
            yield AgentDone(content="done", steps=1)

        with patch("mini_agent.agent.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_stream = fake_run_stream
            instance.add_user_message = MagicMock()

            await tool.execute(prompt="test", tools=["read"])

        call_kwargs = MockAgent.call_args.kwargs
        sub_tool_names = [t.name for t in call_kwargs["tools"]]
        assert sub_tool_names == ["read"]
