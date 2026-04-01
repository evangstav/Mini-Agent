"""JSONL audit log — transcript of all tool calls for session replay and debugging."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".mini-agent" / "sessions"


class AuditLogger:
    """Append-only JSONL writer for tool-call transcripts.

    Each line records: timestamp, tool name, arguments, result summary,
    token usage snapshot, and wall-clock duration.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._dir = SESSIONS_DIR / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "transcript.jsonl"
        self._pending: dict[str, float] = {}  # tool_call_id → start monotonic time
        self._file = open(self._path, "a")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tool_start(self, tool_call_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record the start of a tool call (captures wall-clock start)."""
        self._pending[tool_call_id] = time.monotonic()

    def tool_end(
        self,
        tool_call_id: str,
        tool_name: str,
        *,
        success: bool,
        result_summary: str,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Write a completed tool-call record to the transcript."""
        start = self._pending.pop(tool_call_id, None)
        duration_ms = round((time.monotonic() - start) * 1000) if start is not None else None

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "args": arguments or {},
            "success": success,
            "result": _truncate(result_summary, 500),
            "error": error,
            "duration_ms": duration_ms,
            "tokens": token_usage,
        }

        line = json.dumps(record, separators=(",", ":"), default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
