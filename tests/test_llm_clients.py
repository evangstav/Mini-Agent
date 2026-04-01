"""Test cases for individual Anthropic and OpenAI LLM clients.

These tests require MINIMAX_API_KEY env var to be set.
"""

import os

import pytest

from mini_agent.llm import AnthropicClient, OpenAIClient
from mini_agent.retry import RetryConfig
from mini_agent.schema import Message


def _get_api_key():
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        pytest.skip("MINIMAX_API_KEY not set")
    return key


@pytest.mark.asyncio
async def test_anthropic_simple_completion():
    """Test Anthropic client with simple completion."""
    api_key = _get_api_key()

    client = AnthropicClient(
        api_key=api_key,
        api_base="https://api.minimaxi.com/anthropic",
        model="MiniMax-M2.5",
        retry_config=RetryConfig(enabled=True, max_retries=2),
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello' and nothing else."),
    ]

    response = await client.generate(messages=messages)
    assert response.content
    assert "hello" in response.content.lower()


@pytest.mark.asyncio
async def test_openai_simple_completion():
    """Test OpenAI client with simple completion."""
    api_key = _get_api_key()

    client = OpenAIClient(
        api_key=api_key,
        api_base="https://api.minimaxi.com/v1",
        model="MiniMax-M2.5",
        retry_config=RetryConfig(enabled=True, max_retries=2),
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello' and nothing else."),
    ]

    response = await client.generate(messages=messages)
    assert response.content
    assert "hello" in response.content.lower()
