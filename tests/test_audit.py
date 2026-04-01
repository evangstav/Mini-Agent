"""Tests for mini_agent.audit — JSONL transcript logger."""

import json
from pathlib import Path

from mini_agent.audit import AuditLogger, _truncate


def test_tool_round_trip(tmp_path, monkeypatch):
    """tool_start + tool_end produces a valid JSONL record."""
    monkeypatch.setattr("mini_agent.audit.SESSIONS_DIR", tmp_path)

    logger = AuditLogger("sess-1")
    logger.tool_start("tc-1", "bash", {"command": "ls"})
    logger.tool_end(
        "tc-1",
        "bash",
        success=True,
        result_summary="file1.py\nfile2.py",
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        arguments={"command": "ls"},
    )
    logger.close()

    transcript = tmp_path / "sess-1" / "transcript.jsonl"
    assert transcript.exists()

    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert len(records) == 1

    rec = records[0]
    assert rec["tool"] == "bash"
    assert rec["tool_call_id"] == "tc-1"
    assert rec["success"] is True
    assert rec["args"] == {"command": "ls"}
    assert rec["result"] == "file1.py\nfile2.py"
    assert rec["error"] is None
    assert isinstance(rec["duration_ms"], int)
    assert rec["duration_ms"] >= 0
    assert rec["tokens"]["prompt_tokens"] == 10


def test_tool_end_without_start(tmp_path, monkeypatch):
    """tool_end without matching tool_start sets duration_ms to None."""
    monkeypatch.setattr("mini_agent.audit.SESSIONS_DIR", tmp_path)

    logger = AuditLogger("sess-2")
    logger.tool_end("tc-x", "read", success=False, result_summary="", error="not found")
    logger.close()

    transcript = tmp_path / "sess-2" / "transcript.jsonl"
    rec = json.loads(transcript.read_text().strip())
    assert rec["duration_ms"] is None
    assert rec["success"] is False
    assert rec["error"] == "not found"


def test_truncate():
    assert _truncate("short", 100) == "short"
    assert _truncate("a" * 600, 500) == "a" * 497 + "..."
    assert len(_truncate("a" * 600, 500)) == 500


def test_multiple_records(tmp_path, monkeypatch):
    """Multiple tool calls produce multiple JSONL lines."""
    monkeypatch.setattr("mini_agent.audit.SESSIONS_DIR", tmp_path)

    logger = AuditLogger("sess-3")
    for i in range(3):
        tid = f"tc-{i}"
        logger.tool_start(tid, "bash", {"command": f"echo {i}"})
        logger.tool_end(tid, "bash", success=True, result_summary=str(i))
    logger.close()

    transcript = tmp_path / "sess-3" / "transcript.jsonl"
    lines = transcript.read_text().strip().splitlines()
    assert len(lines) == 3
