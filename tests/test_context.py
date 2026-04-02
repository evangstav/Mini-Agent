"""Tests for context management — tool result storage, compaction, system prompt."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mini_agent.context import (
    SystemPromptBuilder,
    ToolResultStore,
    compact_messages,
    compute_compact_threshold,
    estimate_tokens,
    extract_handoff_context,
)
from mini_agent.schema import LLMResponse, Message


# --- ToolResultStore ---


class TestToolResultStore:
    def test_small_result_unchanged(self, tmp_path):
        store = ToolResultStore(storage_dir=str(tmp_path), preview_chars=100)
        content = "short result"
        result = store.store_if_large(content, "call_1")
        assert result == content

    def test_large_result_truncated_and_stored(self, tmp_path):
        store = ToolResultStore(storage_dir=str(tmp_path), preview_chars=50)
        content = "x" * 200
        result = store.store_if_large(content, "call_2")

        assert "truncated" in result
        assert "200 chars total" in result
        assert result.startswith("x" * 50)

    def test_large_result_retrievable(self, tmp_path):
        store = ToolResultStore(storage_dir=str(tmp_path), preview_chars=50)
        content = "y" * 200
        store.store_if_large(content, "call_3")

        retrieved = store.retrieve("call_3")
        assert retrieved == content

    def test_retrieve_missing_returns_none(self, tmp_path):
        store = ToolResultStore(storage_dir=str(tmp_path))
        assert store.retrieve("nonexistent") is None

    def test_creates_storage_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        store = ToolResultStore(storage_dir=str(nested))
        assert nested.exists()


# --- Token Estimation ---


class TestEstimateTokens:
    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_simple_messages(self):
        msgs = [Message(role="user", content="hello world")]  # 11 chars -> ~2 tokens
        tokens = estimate_tokens(msgs)
        assert tokens == 2  # 11 // 4

    def test_includes_thinking(self):
        msgs = [
            Message(role="assistant", content="reply", thinking="long thinking " * 10)
        ]
        tokens = estimate_tokens(msgs)
        assert tokens > estimate_tokens(
            [Message(role="assistant", content="reply")]
        )


# --- Compaction Threshold ---


class TestComputeCompactThreshold:
    def test_default_values(self):
        # 200_000 * 0.85 - 20_000 = 150_000
        assert compute_compact_threshold() == 150_000

    def test_custom_window(self):
        # 100_000 * 0.85 - 20_000 = 65_000
        assert compute_compact_threshold(context_window=100_000) == 65_000

    def test_custom_pct(self):
        # 200_000 * 0.90 - 20_000 = 160_000
        assert compute_compact_threshold(compact_threshold_pct=0.90) == 160_000

    def test_custom_reserve(self):
        # 200_000 * 0.85 - 10_000 = 160_000
        assert compute_compact_threshold(compaction_reserve=10_000) == 160_000

    def test_all_custom(self):
        # 128_000 * 0.90 - 15_000 = 100_200
        assert compute_compact_threshold(128_000, 0.90, 15_000) == 100_200


# --- Context Compaction ---


@pytest.mark.asyncio
async def test_compact_below_threshold():
    """Messages below threshold are returned unchanged."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    result = await compact_messages(msgs, AsyncMock(), token_threshold=999_999)
    assert result == msgs


@pytest.mark.asyncio
async def test_compact_summarizes_old_turns():
    """When over threshold, old turns are summarized."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Summary of prior work.", finish_reason="stop"
    )

    # Build messages that exceed a low threshold
    msgs = [Message(role="system", content="sys")]
    for i in range(20):
        msgs.append(Message(role="user", content=f"question {i} " * 50))
        msgs.append(Message(role="assistant", content=f"answer {i} " * 50))

    result = await compact_messages(msgs, mock_llm, token_threshold=100, keep_recent=4)

    # Should have: system + summary + 4 recent
    assert result[0].role == "system"
    assert "Context summary" in result[1].content
    assert len(result) == 6  # system + summary + 4 recent


@pytest.mark.asyncio
async def test_compact_percentage_based():
    """Percentage-based threshold triggers compaction correctly."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Summary.", finish_reason="stop"
    )

    msgs = [Message(role="system", content="sys")]
    for i in range(20):
        msgs.append(Message(role="user", content=f"question {i} " * 50))
        msgs.append(Message(role="assistant", content=f"answer {i} " * 50))

    # Use percentage-based threshold with a tiny context window so it triggers
    result = await compact_messages(
        msgs, mock_llm, keep_recent=4,
        context_window=200, compact_threshold_pct=0.85, compaction_reserve=0,
    )

    assert result[0].role == "system"
    assert "Context summary" in result[1].content
    assert len(result) == 6


@pytest.mark.asyncio
async def test_compact_explicit_threshold_overrides_pct():
    """Explicit token_threshold takes precedence over percentage calculation."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    # Percentage would trigger (tiny window) but explicit threshold is huge
    result = await compact_messages(
        msgs, AsyncMock(), token_threshold=999_999,
        context_window=10, compact_threshold_pct=0.5, compaction_reserve=0,
    )
    assert result == msgs  # No compaction because explicit threshold wins


@pytest.mark.asyncio
async def test_compact_fallback_on_llm_failure():
    """If LLM summarization fails, falls back to naive truncation."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("API error")

    msgs = [Message(role="system", content="sys")]
    for i in range(20):
        msgs.append(Message(role="user", content=f"msg {i} " * 50))
        msgs.append(Message(role="assistant", content=f"reply {i} " * 50))

    result = await compact_messages(msgs, mock_llm, token_threshold=100, keep_recent=4)

    assert "omitted due to context limits" in result[1].content


# --- SystemPromptBuilder ---


class TestSystemPromptBuilder:
    def test_base_prompt_only(self, tmp_path):
        builder = SystemPromptBuilder("You are helpful.", str(tmp_path))
        prompt = builder.build()
        assert "You are helpful." in prompt

    def test_includes_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Rules\nBe concise.")
        builder = SystemPromptBuilder("Base prompt.", str(tmp_path))
        prompt = builder.build()
        assert "Be concise." in prompt
        assert "CLAUDE.md" in prompt

    def test_truncates_large_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("x" * 10000)
        builder = SystemPromptBuilder("Base.", str(tmp_path))
        prompt = builder.build()
        assert "truncated" in prompt

    def test_no_claude_md(self, tmp_path):
        # Create .git to bound the upward walk so it doesn't find repo CLAUDE.md
        (tmp_path / ".git").mkdir()
        # Also patch global CLAUDE.md path so the test doesn't find the user's real file
        fake_global = tmp_path / ".no-claude" / "CLAUDE.md"
        with patch.object(Path, "home", return_value=tmp_path / ".no-home"):
            builder = SystemPromptBuilder("Base.", str(tmp_path))
            prompt = builder.build()
        assert "CLAUDE.md" not in prompt

    def test_includes_git_info(self):
        # Use the actual repo directory (we're in a git repo)
        builder = SystemPromptBuilder("Base.", os.getcwd())
        prompt = builder.build()
        # Should include git info since we're in a repo
        assert "Branch:" in prompt or "Git Context" not in prompt


# --- Handoff Context Extraction ---


@pytest.mark.asyncio
async def test_handoff_extracts_relevant_context():
    """Handoff extracts relevant context from prior conversation."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="- File `src/auth.py` has OAuth token refresh logic\n- Uses Redis for session storage",
        finish_reason="stop",
    )

    messages = [
        Message(role="system", content="You are a coding assistant."),
        Message(role="user", content="Fix the OAuth bug in src/auth.py"),
        Message(role="assistant", content="I found the issue in the token refresh logic."),
        Message(role="user", content="Also check the Redis session store."),
        Message(role="assistant", content="Redis sessions are working correctly."),
    ]

    result = await extract_handoff_context(
        messages, "Add rate limiting to the auth endpoint", mock_llm
    )

    # Should have: system + extracted context + new goal
    assert result[0].role == "system"
    assert result[0].content == "You are a coding assistant."
    assert "Relevant context from prior conversation" in result[1].content
    assert "src/auth.py" in result[1].content
    assert result[-1].content == "Add rate limiting to the auth endpoint"
    assert len(result) == 3


@pytest.mark.asyncio
async def test_handoff_no_relevant_context():
    """When LLM says nothing is relevant, only system + goal are returned."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="NO_RELEVANT_CONTEXT", finish_reason="stop"
    )

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="unrelated topic"),
        Message(role="assistant", content="unrelated answer"),
    ]

    result = await extract_handoff_context(messages, "Build a new feature", mock_llm)

    assert len(result) == 2
    assert result[0].role == "system"
    assert result[1].content == "Build a new feature"


@pytest.mark.asyncio
async def test_handoff_empty_conversation():
    """With no prior conversation, returns system + goal."""
    mock_llm = AsyncMock()

    messages = [Message(role="system", content="sys")]
    result = await extract_handoff_context(messages, "Do something", mock_llm)

    assert len(result) == 2
    assert result[0].content == "sys"
    assert result[1].content == "Do something"
    # LLM should not have been called
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_handoff_custom_system_prompt():
    """Custom system prompt overrides the original."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="- relevant info", finish_reason="stop"
    )

    messages = [
        Message(role="system", content="Old system prompt"),
        Message(role="user", content="prior work"),
        Message(role="assistant", content="prior response"),
    ]

    result = await extract_handoff_context(
        messages, "New task", mock_llm, system_prompt="New system prompt"
    )

    assert result[0].content == "New system prompt"


@pytest.mark.asyncio
async def test_handoff_llm_failure_returns_fresh_thread():
    """If extraction LLM fails, returns system + goal without context."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("API error")

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="prior work"),
        Message(role="assistant", content="prior response"),
    ]

    result = await extract_handoff_context(messages, "New task", mock_llm)

    assert len(result) == 2
    assert result[0].content == "sys"
    assert result[1].content == "New task"


@pytest.mark.asyncio
async def test_handoff_truncates_long_extraction():
    """Extracted context is truncated to max_context_chars."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="x" * 5000, finish_reason="stop"
    )

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="prior"),
        Message(role="assistant", content="response"),
    ]

    result = await extract_handoff_context(
        messages, "goal", mock_llm, max_context_chars=100
    )

    context_msg = result[1]
    assert "context truncated" in context_msg.content
    # The content prefix + 100 chars of extraction + truncation notice
    assert len(context_msg.content) < 300


@pytest.mark.asyncio
async def test_handoff_no_system_message_uses_default():
    """If no system message in history, uses a default."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        content="NO_RELEVANT_CONTEXT", finish_reason="stop"
    )

    messages = [
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
    ]

    result = await extract_handoff_context(messages, "New task", mock_llm)

    assert result[0].role == "system"
    assert result[0].content == "You are a helpful assistant."
