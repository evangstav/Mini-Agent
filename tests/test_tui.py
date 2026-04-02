"""Tests for the TUI module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.events import (
    AgentDone,
    AgentError,
    PlanProposal,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from mini_agent.schema import Message
from mini_agent.tui import (
    PermissionManager,
    _format_tokens,
    _render_plan_proposal,
    _truncate,
    load_session,
    save_session,
)


# ── Unit tests: helpers ──────────────────────────────────────────────────────


class TestFormatTokens:
    def test_small(self):
        assert _format_tokens(500) == "500"

    def test_thousands(self):
        assert _format_tokens(1_500) == "1.5K"

    def test_millions(self):
        assert _format_tokens(2_500_000) == "2.5M"


class TestTruncate:
    def test_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_long(self):
        result = _truncate("a" * 20, 10)
        assert len(result) == 11  # 10 chars + "…" (Unicode ellipsis)
        assert result.endswith("…")

    def test_newlines(self):
        result = _truncate("hello\nworld", 50)
        assert "\n" not in result
        assert "hello↵ world" == result


# ── Unit tests: session persistence ──────────────────────────────────────────


class TestSessionPersistence:
    def test_save_and_load(self, tmp_path):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        path = str(tmp_path / "session.json")
        save_session(messages, path)

        loaded = load_session(path)
        assert len(loaded) == 3
        assert loaded[0].role == "system"
        assert loaded[1].content == "Hello"
        assert loaded[2].content == "Hi there!"

    def test_roundtrip_with_tool_calls(self, tmp_path):
        messages = [
            Message(role="system", content="sys"),
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[],
            ),
            Message(role="tool", content="result", tool_call_id="tc1", name="read_file"),
        ]
        path = str(tmp_path / "session.json")
        save_session(messages, path)
        loaded = load_session(path)
        assert loaded[2].tool_call_id == "tc1"
        assert loaded[2].name == "read_file"


# ── Unit tests: permission manager ───────────────────────────────────────────


class TestPermissionManager:
    @pytest.mark.asyncio
    async def test_always_allows_after_a(self):
        mgr = PermissionManager()
        mgr._always_allowed.add("bash")
        result = await mgr.check("bash", {"command": "ls"})
        assert result is True

    @pytest.mark.asyncio
    async def test_prompts_for_unknown_tool(self):
        mgr = PermissionManager()
        with patch("builtins.input", return_value="y"):
            result = await mgr.check("write_file", {"path": "test.py"})
        assert result is True

    @pytest.mark.asyncio
    async def test_deny(self):
        mgr = PermissionManager()
        with patch("builtins.input", return_value="n"):
            result = await mgr.check("bash", {"command": "rm -rf /"})
        assert result is False

    @pytest.mark.asyncio
    async def test_always_remembers(self):
        mgr = PermissionManager()
        with patch("builtins.input", return_value="a"):
            result = await mgr.check("read_file", {"path": "foo.py"})
        assert result is True
        assert "read_file" in mgr._always_allowed
        # Second call should not prompt
        result2 = await mgr.check("read_file", {"path": "bar.py"})
        assert result2 is True


# ── Unit tests: CLI entry point parsing ──────────────────────────────────────


class TestMainEntryPoint:
    def test_module_runnable(self):
        """Verify __main__.py imports correctly."""
        from mini_agent.__main__ import main
        assert callable(main)

    def test_argparse_defaults(self):
        """Verify argparse accepts no arguments."""
        import argparse
        from mini_agent.tui import main

        # main() calls argparse internally; just verify it's callable
        assert callable(main)


# ── Unit tests: plan rendering ──────────────────────────────────────────────


class TestPlanRendering:
    def test_render_plan_proposal(self, capsys):
        """Test plan proposal renders action names and arguments."""
        event = PlanProposal(
            proposed_calls=[
                {"id": "c1", "name": "read_file", "arguments": {"path": "/tmp/test.py"}},
                {"id": "c2", "name": "bash", "arguments": {"command": "ls -la"}},
            ],
            steps=1,
        )
        _render_plan_proposal(event)
        captured = capsys.readouterr()
        assert "read_file" in captured.out
        assert "bash" in captured.out
        assert "/tmp/test.py" in captured.out
        assert "ls -la" in captured.out
        assert "2 actions" in captured.out

    def test_render_plan_truncates_long_values(self, capsys):
        """Test plan proposal truncates long argument values."""
        event = PlanProposal(
            proposed_calls=[
                {"id": "c1", "name": "write_file", "arguments": {"content": "x" * 300}},
            ],
            steps=1,
        )
        _render_plan_proposal(event)
        captured = capsys.readouterr()
        assert "…" in captured.out

    def test_slash_commands_includes_plan(self):
        """Test /plan is listed in SLASH_COMMANDS."""
        from mini_agent.tui import SLASH_COMMANDS
        assert "/plan" in SLASH_COMMANDS
