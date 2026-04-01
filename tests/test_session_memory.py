"""Tests for session memory — periodic markdown snapshot of conversation state."""

import tempfile
from pathlib import Path

import pytest

from mini_agent.session_memory import SessionMemory


class TestSessionMemory:
    def test_initial_state(self):
        mem = SessionMemory()
        assert mem._task_description == "(not yet identified)"
        assert mem._tool_calls_since_update == 0
        assert len(mem._files_touched) == 0

    def test_record_tool_call_tracks_files(self):
        mem = SessionMemory()
        mem.record_tool_call("Read", {"file_path": "/foo/bar.py"}, "content", True)
        assert "/foo/bar.py" in mem._files_touched

    def test_record_tool_call_tracks_bash(self):
        mem = SessionMemory()
        mem.record_tool_call("Bash", {"command": "ls -la"}, "output", True)
        assert len(mem._worklog) == 1
        assert "`ls -la`" in mem._worklog[0]

    def test_record_tool_call_tracks_errors(self):
        mem = SessionMemory()
        mem.record_tool_call("Bash", {"command": "bad cmd"}, "error msg", False)
        assert len(mem._errors) == 1
        assert "FAILED" in mem._worklog[0]

    def test_record_tool_call_non_bash_error(self):
        mem = SessionMemory()
        mem.record_tool_call("Read", {"file_path": "/x"}, "not found", False)
        assert len(mem._errors) == 1
        assert "Read" in mem._errors[0]

    def test_record_assistant_text_sets_task(self):
        mem = SessionMemory()
        mem.record_assistant_text("I will implement the session memory feature for the agent.")
        assert "session memory" in mem._task_description

    def test_record_assistant_text_ignores_short(self):
        mem = SessionMemory()
        mem.record_assistant_text("ok")
        assert mem._task_description == "(not yet identified)"

    def test_record_assistant_text_only_first(self):
        mem = SessionMemory()
        mem.record_assistant_text("First task description that is long enough")
        mem.record_assistant_text("Second completely different task description")
        assert "First" in mem._task_description

    def test_should_update_by_calls(self):
        mem = SessionMemory(update_interval_calls=3)
        mem._tool_calls_since_update = 2
        assert not mem.should_update(0)
        mem._tool_calls_since_update = 3
        assert mem.should_update(0)

    def test_should_update_by_tokens(self):
        mem = SessionMemory(update_interval_tokens=1000)
        mem._tokens_at_last_update = 5000
        assert not mem.should_update(5500)
        assert mem.should_update(6000)

    def test_write_snapshot(self, tmp_path):
        output = tmp_path / "session_memory.md"
        mem = SessionMemory(output_path=output)
        mem.record_tool_call("Bash", {"command": "echo hello"}, "hello", True)
        mem.record_tool_call("Read", {"file_path": "/src/main.py"}, "code", True)
        mem.update_state("Implementing feature")
        mem.add_learning("Use dataclasses not dicts")

        path = mem.write_snapshot(current_token_estimate=5000)
        assert path == output
        assert output.exists()

        content = output.read_text()
        assert "# Session Memory" in content
        assert "Implementing feature" in content
        assert "/src/main.py" in content
        assert "echo hello" in content
        assert "Use dataclasses not dicts" in content

    def test_write_snapshot_resets_counters(self):
        mem = SessionMemory(output_path=Path(tempfile.mktemp(suffix=".md")))
        mem._tool_calls_since_update = 15
        mem.write_snapshot(current_token_estimate=10000)
        assert mem._tool_calls_since_update == 0
        assert mem._tokens_at_last_update == 10000
        # Clean up
        mem.output_path.unlink(missing_ok=True)

    def test_as_compaction_summary_returns_none_when_sparse(self):
        mem = SessionMemory()
        assert mem.as_compaction_summary() is None

    def test_as_compaction_summary_returns_content(self):
        mem = SessionMemory()
        mem.record_tool_call("Bash", {"command": "make build"}, "ok", True)
        mem.record_assistant_text("Implementing the build system for the project.")
        summary = mem.as_compaction_summary()
        assert summary is not None
        assert "# Session Memory" in summary
        assert "make build" in summary

    def test_update_state(self):
        mem = SessionMemory()
        mem.update_state("Running tests")
        assert mem._current_state == "Running tests"

    def test_format_list_empty(self):
        assert SessionMemory._format_list([]) == ""

    def test_format_list_items(self):
        result = SessionMemory._format_list(["a", "b"])
        assert result == "- a\n- b"

    def test_write_snapshot_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "deep" / "nested" / "session_memory.md"
        mem = SessionMemory(output_path=output)
        mem.record_tool_call("Bash", {"command": "ls"}, "files", True)
        mem.write_snapshot()
        assert output.exists()

    def test_glob_tool_tracks_pattern(self):
        mem = SessionMemory()
        mem.record_tool_call("Glob", {"pattern": "**/*.py"}, "files", True)
        assert "**/*.py" in mem._files_touched

    def test_grep_tool_tracks_path(self):
        mem = SessionMemory()
        mem.record_tool_call("Grep", {"path": "/src", "pattern": "TODO"}, "matches", True)
        assert "/src" in mem._files_touched

    def test_long_bash_command_truncated_in_worklog(self):
        mem = SessionMemory()
        long_cmd = "x" * 200
        mem.record_tool_call("Bash", {"command": long_cmd}, "ok", True)
        # Worklog entry should be truncated at 120 chars
        assert len(mem._worklog[0]) < 200
