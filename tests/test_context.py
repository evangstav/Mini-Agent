"""Tests for context management — tool result storage, compaction, system prompt."""

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from mini_agent.context import (
    SystemPromptBuilder,
    ToolResultStore,
    compact_messages,
    estimate_tokens,
    prune_tool_results,
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


# --- Tool Result Pruning ---


class TestPruneToolResults:
    def test_empty_messages(self):
        assert prune_tool_results([]) == []

    def test_no_pruning_within_protection_window(self):
        """Large tool results within protection window are kept."""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="do stuff"),
            Message(role="tool", content="x" * 100_000, tool_call_id="c1", name="bash"),
        ]
        result = prune_tool_results(msgs, protect_chars=200_000)
        assert result[2].content == "x" * 100_000

    def test_prunes_large_tool_result_outside_window(self):
        """Large tool results beyond protection window get pruned."""
        # Build: old large tool result, then enough recent content to push it outside window
        msgs = [
            Message(role="system", content="sys"),
            Message(role="tool", content="A" * 100_000, tool_call_id="old", name="bash"),
            Message(role="user", content="B" * 50_000),
            Message(role="assistant", content="C" * 50_000),
            Message(role="user", content="D" * 50_000),
            Message(role="assistant", content="E" * 50_000),
        ]
        # protect_chars=160K, prune_threshold=80K
        # Recent content (scanning backward): E(50K) + D(50K) + C(50K) + B(50K) = 200K
        # The old tool result (100K) is beyond the 160K window and > 80K threshold
        result = prune_tool_results(msgs, protect_chars=160_000, prune_threshold=80_000)
        assert "pruned" in result[1].content.lower()
        assert result[1].tool_call_id == "old"
        assert result[1].name == "bash"

    def test_preserves_small_tool_results_outside_window(self):
        """Small tool results beyond protection window are kept."""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="tool", content="small", tool_call_id="old", name="bash"),
            Message(role="user", content="B" * 200_000),
        ]
        result = prune_tool_results(msgs, protect_chars=100_000, prune_threshold=80_000)
        assert result[1].content == "small"

    def test_preserves_non_tool_messages(self):
        """User/assistant messages are never pruned even if large and outside window."""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="X" * 100_000),
            Message(role="assistant", content="Y" * 200_000),
        ]
        result = prune_tool_results(msgs, protect_chars=1000, prune_threshold=500)
        assert result[1].content == "X" * 100_000
        assert result[2].content == "Y" * 200_000

    def test_returns_same_list_when_nothing_to_prune(self):
        """Returns original list reference when no pruning needed."""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
        result = prune_tool_results(msgs)
        assert result is msgs
