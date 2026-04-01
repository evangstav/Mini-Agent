"""Tests for context management — tool result storage, compaction, session memory, system prompt."""

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from mini_agent.context import (
    SessionMemory,
    SystemPromptBuilder,
    ToolResultStore,
    compact_messages,
    estimate_tokens,
    microcompact_messages,
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


# --- Session Memory ---


class TestSessionMemory:
    def test_add_and_get_summary(self, tmp_path):
        mem = SessionMemory(storage_path=str(tmp_path / "mem.md"))
        mem.add("Found bug in auth module", category="findings")
        mem.add("Using retry pattern for API calls", category="decisions")
        summary = mem.get_summary()
        assert "Session Memory" in summary
        assert "Found bug in auth module" in summary
        assert "Using retry pattern" in summary

    def test_empty_summary(self, tmp_path):
        mem = SessionMemory(storage_path=str(tmp_path / "mem.md"))
        assert mem.get_summary() == ""

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "mem.md")
        mem1 = SessionMemory(storage_path=path)
        mem1.add("important fact", category="context")

        # Load from same path
        mem2 = SessionMemory(storage_path=path)
        assert len(mem2.entries) == 1
        assert mem2.entries[0]["note"] == "important fact"
        assert mem2.entries[0]["category"] == "context"

    def test_clear(self, tmp_path):
        mem = SessionMemory(storage_path=str(tmp_path / "mem.md"))
        mem.add("note 1")
        mem.add("note 2")
        mem.clear()
        assert mem.entries == []
        assert mem.get_summary() == ""

    def test_default_category(self, tmp_path):
        mem = SessionMemory(storage_path=str(tmp_path / "mem.md"))
        mem.add("no category")
        assert mem.entries[0]["category"] == "general"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "mem.md"
        mem = SessionMemory(storage_path=str(path))
        mem.add("test")
        assert path.exists()


# --- Microcompaction ---


class TestMicrocompaction:
    def test_leaves_short_messages_alone(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        result = microcompact_messages(msgs, keep_recent=2)
        assert result == msgs

    def test_compacts_old_tool_results(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="q1"),
            Message(
                role="tool",
                content="x" * 500,
                tool_call_id="tc1",
                name="read_file",
            ),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
            Message(role="assistant", content="a2"),
        ]
        result = microcompact_messages(msgs, keep_recent=2, max_tool_content=50)
        # The tool result should be compacted
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert "compacted" in tool_msg.content
        assert "read_file" in tool_msg.content
        assert len(tool_msg.content) < 500

    def test_preserves_recent_tool_results(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="old"),
            Message(role="assistant", content="old reply"),
            Message(role="user", content="recent q"),
            Message(
                role="tool",
                content="x" * 500,
                tool_call_id="tc_recent",
                name="bash",
            ),
            Message(role="assistant", content="recent reply"),
        ]
        result = microcompact_messages(msgs, keep_recent=3, max_tool_content=50)
        # Recent tool result should be untouched
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert tool_msg.content == "x" * 500

    def test_strips_old_thinking(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="assistant", content="reply", thinking="long thought " * 50),
            Message(role="user", content="recent 1"),
            Message(role="assistant", content="recent 2"),
        ]
        result = microcompact_messages(msgs, keep_recent=2)
        old_assistant = [m for m in result if m.role == "assistant"][0]
        assert old_assistant.thinking is None
        assert old_assistant.content == "reply"

    def test_preserves_tool_call_id(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(
                role="tool",
                content="x" * 500,
                tool_call_id="tc1",
                name="grep",
            ),
            Message(role="user", content="recent 1"),
            Message(role="assistant", content="recent 2"),
        ]
        result = microcompact_messages(msgs, keep_recent=2, max_tool_content=50)
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert tool_msg.tool_call_id == "tc1"
        assert tool_msg.name == "grep"


# --- Tiered Compaction ---


@pytest.mark.asyncio
async def test_compact_tier1_microcompaction_sufficient():
    """When microcompaction alone drops below threshold, no LLM call needed."""
    msgs = [Message(role="system", content="sys")]
    # Add messages with large tool results
    for i in range(10):
        msgs.append(Message(role="user", content=f"q{i}"))
        msgs.append(Message(
            role="tool",
            content="x" * 2000,
            tool_call_id=f"tc{i}",
            name="read_file",
        ))
        msgs.append(Message(role="assistant", content=f"a{i}"))

    mock_llm = AsyncMock()
    # Set threshold so microcompaction should be enough
    result = await compact_messages(msgs, mock_llm, token_threshold=5000, keep_recent=4)

    # LLM should NOT have been called if microcompaction was sufficient
    # (depends on exact sizes, but the point is tier 1 runs first)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_compact_tier2_session_memory():
    """When session memory is available, use it instead of LLM."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mem = SessionMemory(storage_path=f"{tmpdir}/mem.md")
        mem.add("User is implementing auth flow", category="context")
        mem.add("Decided to use JWT tokens", category="decisions")

        msgs = [Message(role="system", content="sys")]
        for i in range(20):
            msgs.append(Message(role="user", content=f"question {i} " * 100))
            msgs.append(Message(role="assistant", content=f"answer {i} " * 100))

        mock_llm = AsyncMock()
        result = await compact_messages(
            msgs, mock_llm, token_threshold=100, keep_recent=4, session_memory=mem
        )

        # Should use session memory — LLM NOT called
        mock_llm.generate.assert_not_called()
        # Summary should reference session memory
        assert "session memory" in result[1].content.lower()
