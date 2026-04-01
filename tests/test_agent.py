"""Test cases for Agent."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mini_agent.agent import Agent
from mini_agent.events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from mini_agent.tools.base import Tool, ToolResult


class MockTool(Tool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=f"Mock result for: {kwargs.get('input', '')}")


# --- Backward-compatible run() tests ---


@pytest.mark.asyncio
async def test_agent_no_tool_calls():
    """Test agent returns immediately when LLM gives no tool calls."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Hello!", finish_reason="stop"
    )

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[])
    agent.add_user_message("Hi")
    result = await agent.run()

    assert result == "Hello!"
    assert mock_llm.generate.call_count == 1


@pytest.mark.asyncio
async def test_agent_with_tool_call():
    """Test agent handles tool calls correctly."""
    mock_llm = AsyncMock()

    # First call: LLM wants to use a tool
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="Let me use the tool.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
                )
            ],
            finish_reason="tool_use",
        ),
        # Second call: LLM gives final answer
        LLMResponse(content="Done! The result was mock.", finish_reason="stop"),
    ]

    tool = MockTool()
    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[tool])
    agent.add_user_message("Use the mock tool")
    result = await agent.run()

    assert result == "Done! The result was mock."
    assert mock_llm.generate.call_count == 2


@pytest.mark.asyncio
async def test_agent_unknown_tool():
    """Test agent handles unknown tool gracefully."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="nonexistent_tool", arguments={}),
                )
            ],
            finish_reason="tool_use",
        ),
        LLMResponse(content="Tool not found.", finish_reason="stop"),
    ]

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[])
    agent.add_user_message("Do something")
    result = await agent.run()

    assert result == "Tool not found."


@pytest.mark.asyncio
async def test_agent_cancellation():
    """Test agent respects cancel event."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="This should not be the result",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
            )
        ],
        finish_reason="tool_use",
    )

    cancel_event = asyncio.Event()
    cancel_event.set()  # Cancel immediately

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[MockTool()])
    agent.add_user_message("Do something")
    result = await agent.run(cancel_event=cancel_event)

    assert "cancelled" in result.lower()


@pytest.mark.asyncio
async def test_agent_max_steps():
    """Test agent stops at max steps."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Still working...",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(name="mock_tool", arguments={"input": "loop"}),
            )
        ],
        finish_reason="tool_use",
    )

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[MockTool()], max_steps=3)
    agent.add_user_message("Loop forever")
    result = await agent.run()

    assert "couldn't be completed" in result.lower()
    assert mock_llm.generate.call_count == 3


@pytest.mark.asyncio
async def test_agent_get_history():
    """Test get_history returns message copy."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(content="Hi!", finish_reason="stop")

    agent = Agent(llm_client=mock_llm, system_prompt="System", tools=[])
    agent.add_user_message("Hello")
    await agent.run()

    history = agent.get_history()
    assert len(history) == 3  # system + user + assistant
    assert history[0].role == "system"
    assert history[1].role == "user"
    assert history[2].role == "assistant"


# --- Streaming run_stream() tests ---


@pytest.mark.asyncio
async def test_stream_no_tool_calls():
    """Test run_stream yields TextChunk and AgentDone when no tools used."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Hello!", finish_reason="stop"
    )

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[])
    agent.add_user_message("Hi")

    events = [event async for event in agent.run_stream()]

    assert any(isinstance(e, TextChunk) and e.content == "Hello!" for e in events)
    assert isinstance(events[-1], AgentDone)
    assert events[-1].content == "Hello!"
    assert events[-1].steps == 1


@pytest.mark.asyncio
async def test_stream_with_thinking():
    """Test run_stream yields ThinkingChunk when thinking is present."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="The answer is 42.",
        thinking="Let me think about this...",
        finish_reason="stop",
    )

    agent = Agent(llm_client=mock_llm, system_prompt="You are helpful.", tools=[])
    agent.add_user_message("What is the meaning?")

    events = [event async for event in agent.run_stream()]

    thinking_events = [e for e in events if isinstance(e, ThinkingChunk)]
    assert len(thinking_events) == 1
    assert thinking_events[0].content == "Let me think about this..."

    text_events = [e for e in events if isinstance(e, TextChunk)]
    assert len(text_events) == 1
    assert text_events[0].content == "The answer is 42."


@pytest.mark.asyncio
async def test_stream_tool_lifecycle():
    """Test run_stream yields ToolStart and ToolEnd around tool execution."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="Using tool.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
                )
            ],
            finish_reason="tool_use",
        ),
        LLMResponse(content="Done.", finish_reason="stop"),
    ]

    tool = MockTool()
    agent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[tool])
    agent.add_user_message("Use the tool")

    events = [event async for event in agent.run_stream()]

    # Check ToolStart
    starts = [e for e in events if isinstance(e, ToolStart)]
    assert len(starts) == 1
    assert starts[0].tool_name == "mock_tool"
    assert starts[0].tool_call_id == "call_1"
    assert starts[0].arguments == {"input": "test"}

    # Check ToolEnd
    ends = [e for e in events if isinstance(e, ToolEnd)]
    assert len(ends) == 1
    assert ends[0].tool_name == "mock_tool"
    assert ends[0].success is True
    assert "Mock result" in ends[0].content

    # ToolStart comes before ToolEnd
    start_idx = events.index(starts[0])
    end_idx = events.index(ends[0])
    assert start_idx < end_idx


@pytest.mark.asyncio
async def test_stream_unknown_tool():
    """Test run_stream yields ToolEnd with error for unknown tools."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="nonexistent", arguments={}),
                )
            ],
            finish_reason="tool_use",
        ),
        LLMResponse(content="OK", finish_reason="stop"),
    ]

    agent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    agent.add_user_message("Do it")

    events = [event async for event in agent.run_stream()]

    ends = [e for e in events if isinstance(e, ToolEnd)]
    assert len(ends) == 1
    assert ends[0].success is False
    assert "Unknown tool" in ends[0].error


@pytest.mark.asyncio
async def test_stream_cancellation():
    """Test run_stream yields AgentDone on cancellation."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Working...",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
            )
        ],
        finish_reason="tool_use",
    )

    cancel_event = asyncio.Event()
    cancel_event.set()

    agent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[MockTool()])
    agent.add_user_message("Go")

    events = [event async for event in agent.run_stream(cancel_event=cancel_event)]

    assert isinstance(events[-1], AgentCancelled)
    assert "cancelled" in events[-1].content.lower()


@pytest.mark.asyncio
async def test_stream_max_steps():
    """Test run_stream yields AgentError when max steps exceeded."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Working...",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(name="mock_tool", arguments={"input": "loop"}),
            )
        ],
        finish_reason="tool_use",
    )

    agent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[MockTool()], max_steps=2)
    agent.add_user_message("Loop")

    events = [event async for event in agent.run_stream()]

    assert isinstance(events[-1], AgentError)
    assert "couldn't be completed" in events[-1].error.lower()
    assert events[-1].steps == 2


@pytest.mark.asyncio
async def test_stream_llm_error():
    """Test run_stream yields AgentError on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("API down")

    agent = Agent(llm_client=mock_llm, system_prompt="Test", tools=[])
    agent.add_user_message("Hi")

    events = [event async for event in agent.run_stream()]

    assert len(events) == 1
    assert isinstance(events[0], AgentError)
    assert "LLM call failed" in events[0].error


@pytest.mark.asyncio
async def test_stream_permission_denied():
    """Test run_stream handles permission denial."""

    async def deny_all(tool_name, arguments):
        return False

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="Let me run this.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
                )
            ],
            finish_reason="tool_use",
        ),
        LLMResponse(content="Permission was denied.", finish_reason="stop"),
    ]

    tool = MockTool()
    agent = Agent(
        llm_client=mock_llm,
        system_prompt="Test",
        tools=[tool],
        permission_callback=deny_all,
    )
    agent.add_user_message("Use the tool")

    events = [event async for event in agent.run_stream()]

    # Should see PermissionRequest, then ToolEnd with denied
    perm_events = [e for e in events if isinstance(e, PermissionRequest)]
    assert len(perm_events) == 1
    assert perm_events[0].tool_name == "mock_tool"

    # No ToolStart since permission was denied
    starts = [e for e in events if isinstance(e, ToolStart)]
    assert len(starts) == 0

    # ToolEnd with denial error
    ends = [e for e in events if isinstance(e, ToolEnd)]
    assert len(ends) == 1
    assert ends[0].success is False
    assert "Permission denied" in ends[0].error


@pytest.mark.asyncio
async def test_stream_permission_granted():
    """Test run_stream proceeds when permission is granted."""

    async def allow_all(tool_name, arguments):
        return True

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        LLMResponse(
            content="Running tool.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="mock_tool", arguments={"input": "test"}),
                )
            ],
            finish_reason="tool_use",
        ),
        LLMResponse(content="Done.", finish_reason="stop"),
    ]

    tool = MockTool()
    agent = Agent(
        llm_client=mock_llm,
        system_prompt="Test",
        tools=[tool],
        permission_callback=allow_all,
    )
    agent.add_user_message("Use the tool")

    events = [event async for event in agent.run_stream()]

    # Should see PermissionRequest, ToolStart, ToolEnd
    perm_events = [e for e in events if isinstance(e, PermissionRequest)]
    assert len(perm_events) == 1

    starts = [e for e in events if isinstance(e, ToolStart)]
    assert len(starts) == 1

    ends = [e for e in events if isinstance(e, ToolEnd)]
    assert len(ends) == 1
    assert ends[0].success is True
