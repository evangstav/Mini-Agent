"""Test cases for Agent."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mini_agent.agent import Agent
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
