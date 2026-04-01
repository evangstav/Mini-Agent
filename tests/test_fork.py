"""Tests for sub-agent spawning via Agent.fork()."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mini_agent.agent import Agent
from mini_agent.events import SubAgentDone, SubAgentError, SubAgentSpawn
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from mini_agent.tools.base import Tool, ToolResult


class EchoTool(Tool):
    """Simple tool that echoes input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes input"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=kwargs.get("text", ""))


# --- fork() tests ---


@pytest.mark.asyncio
async def test_fork_creates_isolated_agent():
    """Forked agent has independent message history."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(content="Hi!", finish_reason="stop")

    parent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[])
    parent.add_user_message("Parent message")

    child = parent.fork("Child task")

    # Child should have parent's messages plus the new task
    assert len(child.messages) == len(parent.messages) + 1
    assert child.messages[-1].content == "Child task"
    assert child.messages[-1].role == "user"

    # Mutating child should not affect parent
    child.add_user_message("Extra child message")
    assert len(child.messages) == len(parent.messages) + 2


@pytest.mark.asyncio
async def test_fork_shares_system_prompt():
    """Forked agent uses byte-identical system prompt for cache hits."""
    mock_llm = AsyncMock()
    system = "You are a precise assistant."

    parent = Agent(llm_client=mock_llm, system_prompt=system, tools=[])
    child = parent.fork("Do something")

    assert child.system_prompt is parent.system_prompt
    assert child.system_prompt == system


@pytest.mark.asyncio
async def test_fork_shares_llm_client():
    """Forked agent shares the same LLM client instance (prompt cache)."""
    mock_llm = AsyncMock()
    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    child = parent.fork("Task")

    assert child.llm is parent.llm


@pytest.mark.asyncio
async def test_fork_inherits_tools():
    """Forked agent inherits parent's tools by default."""
    mock_llm = AsyncMock()
    tool = EchoTool()
    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[tool])
    child = parent.fork("Task")

    assert "echo" in child.tools
    assert len(child.tools) == 1


@pytest.mark.asyncio
async def test_fork_overrides_tools():
    """Forked agent can use different tools."""
    mock_llm = AsyncMock()
    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[EchoTool()])

    class OtherTool(Tool):
        @property
        def name(self):
            return "other"

        @property
        def description(self):
            return "Other"

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return ToolResult(success=True, content="other")

    child = parent.fork("Task", tools=[OtherTool()])

    assert "other" in child.tools
    assert "echo" not in child.tools


@pytest.mark.asyncio
async def test_fork_overrides_max_steps():
    """Forked agent can have different max_steps."""
    mock_llm = AsyncMock()
    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[], max_steps=50)
    child = parent.fork("Task", max_steps=5)

    assert child.max_steps == 5
    assert parent.max_steps == 50


@pytest.mark.asyncio
async def test_fork_runs_independently():
    """Forked agent can complete a task independently."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Sub-task done!", finish_reason="stop"
    )

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    child = parent.fork("Do the sub-task")
    result = await child.run()

    assert result == "Sub-task done!"


@pytest.mark.asyncio
async def test_fork_inherits_config():
    """Forked agent inherits compaction and context config."""
    mock_llm = AsyncMock()
    parent = Agent(
        llm_client=mock_llm,
        system_prompt="Test",
        tools=[],
        compact_threshold=5000,
        context_window=100_000,
        compact_threshold_pct=0.9,
        compaction_reserve=10_000,
    )
    child = parent.fork("Task")

    assert child.compact_threshold == parent.compact_threshold
    assert child.context_window == parent.context_window
    assert child.compact_threshold_pct == parent.compact_threshold_pct
    assert child.compaction_reserve == parent.compaction_reserve


# --- run_forked() tests ---


@pytest.mark.asyncio
async def test_run_forked_parallel():
    """run_forked runs multiple tasks concurrently and returns results."""
    call_count = 0

    async def mock_generate(messages, tools=None):
        nonlocal call_count
        call_count += 1
        # Find the last user message to identify the task
        user_msgs = [m for m in messages if m.role == "user"]
        task = user_msgs[-1].content if user_msgs else "unknown"
        return LLMResponse(content=f"Result for: {task}", finish_reason="stop")

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = mock_generate

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    parent.add_user_message("Initial context")

    results = await parent.run_forked(["Task A", "Task B", "Task C"])

    assert len(results) == 3
    result_texts = [r[1] for r in results]
    assert "Result for: Task A" in result_texts
    assert "Result for: Task B" in result_texts
    assert "Result for: Task C" in result_texts


@pytest.mark.asyncio
async def test_run_forked_handles_errors():
    """run_forked handles sub-agent errors gracefully."""
    call_order = 0

    async def mock_generate(messages, tools=None):
        nonlocal call_order
        call_order += 1
        user_msgs = [m for m in messages if m.role == "user"]
        task = user_msgs[-1].content if user_msgs else ""
        if "fail" in task.lower():
            raise RuntimeError("Simulated failure")
        return LLMResponse(content=f"OK: {task}", finish_reason="stop")

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = mock_generate

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])

    results = await parent.run_forked(["Good task", "Fail task"])

    assert len(results) == 2
    # One succeeds, one fails (Agent.run() catches LLM errors internally)
    result_texts = [r[1] for r in results]
    successes = [t for t in result_texts if t.startswith("OK:")]
    failures = [t for t in result_texts if "failed" in t.lower() or "error" in t.lower()]
    assert len(successes) == 1
    assert len(failures) == 1


@pytest.mark.asyncio
async def test_run_forked_empty_tasks():
    """run_forked with no tasks returns empty list."""
    mock_llm = AsyncMock()
    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])

    results = await parent.run_forked([])
    assert results == []


@pytest.mark.asyncio
async def test_run_forked_does_not_modify_parent():
    """run_forked does not add messages to parent's history."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(content="Done", finish_reason="stop")

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    parent.add_user_message("Context")
    parent_msg_count = len(parent.messages)

    await parent.run_forked(["Sub-task 1", "Sub-task 2"])

    assert len(parent.messages) == parent_msg_count


# --- run_forked_stream() tests ---


@pytest.mark.asyncio
async def test_run_forked_stream_events():
    """run_forked_stream yields spawn and done events."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(content="Done", finish_reason="stop")

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])

    events = [event async for event in parent.run_forked_stream(["Task A", "Task B"])]

    spawns = [e for e in events if isinstance(e, SubAgentSpawn)]
    dones = [e for e in events if isinstance(e, SubAgentDone)]

    assert len(spawns) == 2
    assert len(dones) == 2
    assert {s.task for s in spawns} == {"Task A", "Task B"}


@pytest.mark.asyncio
async def test_run_forked_stream_error_events():
    """run_forked_stream yields error events for failed sub-agents."""
    async def mock_generate(messages, tools=None):
        user_msgs = [m for m in messages if m.role == "user"]
        task = user_msgs[-1].content if user_msgs else ""
        if "fail" in task.lower():
            raise RuntimeError("Boom")
        return LLMResponse(content="OK", finish_reason="stop")

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = mock_generate

    parent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])

    events = [event async for event in parent.run_forked_stream(["Good", "Fail here"])]

    spawns = [e for e in events if isinstance(e, SubAgentSpawn)]
    dones = [e for e in events if isinstance(e, SubAgentDone)]

    assert len(spawns) == 2
    # Agent.run() catches LLM errors internally, so both come back as SubAgentDone
    assert len(dones) == 2
    # One result is a success, one contains the error
    results = [d.result for d in dones]
    assert any("OK" in r for r in results)
    assert any("failed" in r.lower() or "Boom" in r for r in results)
