"""Test cases for LLM wrapper client."""

import os

import pytest

from mini_agent.llm import LLMClient
from mini_agent.schema import LLMProvider, Message


def _get_api_key():
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        pytest.skip("MINIMAX_API_KEY not set")
    return key


def test_wrapper_default_provider():
    """Test LLM wrapper defaults to Anthropic provider."""
    client = LLMClient(api_key="test-key", model="test-model")
    assert client.provider == LLMProvider.ANTHROPIC


def test_wrapper_openai_provider():
    """Test LLM wrapper with OpenAI provider."""
    client = LLMClient(api_key="test-key", provider=LLMProvider.OPENAI, model="test-model")
    assert client.provider == LLMProvider.OPENAI


@pytest.mark.asyncio
async def test_wrapper_anthropic_generate():
    """Test Anthropic provider generation (requires API key)."""
    api_key = _get_api_key()
    client = LLMClient(api_key=api_key, provider=LLMProvider.ANTHROPIC)

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello' and nothing else."),
    ]

    response = await client.generate(messages=messages)
    assert response.content
    assert "hello" in response.content.lower()
