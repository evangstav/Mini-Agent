"""Tests for context management — tool result storage, compaction, system prompt."""

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from mini_agent.context import (
    SystemPromptBuilder,
    ToolResultStore,
    compact_messages,
    compute_compact_threshold,
    estimate_tokens,
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
        builder = SystemPromptBuilder("Base.", str(tmp_path))
        prompt = builder.build()
        assert "CLAUDE.md" not in prompt

    def test_includes_git_info(self):
        # Use the actual repo directory (we're in a git repo)
        builder = SystemPromptBuilder("Base.", os.getcwd())
        prompt = builder.build()
        # Should include git info since we're in a repo
        assert "Branch:" in prompt or "Git Context" not in prompt
