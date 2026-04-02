"""Session memory — periodic markdown snapshot of conversation state.

Maintains a running session_memory.md file that captures the conversation's
current state, task context, files touched, errors, and learnings. Updated
every N tool calls or N tokens of context growth.

Integrates with compaction: the session memory file can serve as a free
summary (no LLM call) when context compaction is triggered.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .schema import Message

logger = logging.getLogger(__name__)

# Template sections for the session memory file
_TEMPLATE = """\
# Session Memory

_Auto-updated: {timestamp}_

## Current State
{current_state}

## Task
{task}

## Files Touched
{files}

## Workflow Progress
{workflow}

## Errors & Blockers
{errors}

## Learnings & Decisions
{learnings}

## Work Log
{worklog}
"""


class SessionMemory:
    """Tracks conversation state and writes periodic markdown snapshots.

    Args:
        output_path: Where to write session_memory.md.
        update_interval_calls: Write snapshot every N tool calls.
        update_interval_tokens: Write snapshot after N tokens of growth.
    """

    def __init__(
        self,
        output_path: str | Path = ".runtime/session_memory.md",
        update_interval_calls: int = 10,
        update_interval_tokens: int = 8_000,
    ):
        self.output_path = Path(output_path)
        self.update_interval_calls = update_interval_calls
        self.update_interval_tokens = update_interval_tokens

        # Counters
        self._tool_calls_since_update = 0
        self._tokens_at_last_update = 0

        # Accumulated state
        self._task_description: str = "(not yet identified)"
        self._files_touched: set[str] = set()
        self._errors: list[str] = []
        self._learnings: list[str] = []
        self._worklog: list[str] = []
        self._current_state: str = "Starting"

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_content: str,
        success: bool,
    ) -> None:
        """Record a tool call and extract relevant state."""
        self._tool_calls_since_update += 1

        # Track files from file-oriented tools
        path = arguments.get("file_path") or arguments.get("path") or arguments.get("pattern")
        if path and tool_name in (
            "read_file", "write_file", "edit_file", "glob", "grep",
            "Read", "Write", "Edit", "Glob", "Grep",
        ):
            self._files_touched.add(str(path))

        # Track bash commands
        if tool_name in ("bash", "Bash") and "command" in arguments:
            cmd = arguments["command"]
            entry = f"`{cmd[:120]}`"
            if not success:
                entry += " **FAILED**"
                error_snippet = result_content[:200] if result_content else "(no output)"
                self._errors.append(f"{tool_name}: {error_snippet}")
            self._worklog.append(entry)

        # Track errors from any tool
        if not success and tool_name not in ("bash", "Bash"):
            snippet = result_content[:200] if result_content else "(no output)"
            self._errors.append(f"{tool_name}: {snippet}")

    def record_assistant_text(self, text: str) -> None:
        """Extract task description from early assistant messages."""
        if self._task_description == "(not yet identified)" and len(text) > 20:
            # Use first substantial assistant message as task description
            self._task_description = text[:300].strip()

    def update_state(self, state: str) -> None:
        """Set the current state label."""
        self._current_state = state

    def add_learning(self, learning: str) -> None:
        """Record a decision or learning."""
        self._learnings.append(learning)

    def should_update(self, current_token_estimate: int) -> bool:
        """Check if it's time to write a snapshot."""
        if self._tool_calls_since_update >= self.update_interval_calls:
            return True
        token_growth = current_token_estimate - self._tokens_at_last_update
        if token_growth >= self.update_interval_tokens:
            return True
        return False

    def write_snapshot(self, current_token_estimate: int = 0) -> Path:
        """Write the current session memory to disk.

        Returns the path written to.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        content = _TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            current_state=self._current_state,
            task=self._task_description,
            files=self._format_list(sorted(self._files_touched)) or "(none)",
            workflow=self._format_list(self._workflow_summary()) or "(no progress yet)",
            errors=self._format_list(self._errors[-10:]) or "(none)",
            learnings=self._format_list(self._learnings[-10:]) or "(none)",
            worklog=self._format_list(self._worklog[-30:]) or "(empty)",
        )

        self.output_path.write_text(content, encoding="utf-8")

        # Reset counters
        self._tool_calls_since_update = 0
        self._tokens_at_last_update = current_token_estimate

        logger.debug("Session memory snapshot written to %s", self.output_path)
        return self.output_path

    def as_compaction_summary(self) -> str | None:
        """Return session memory content as a compaction summary.

        This is the "free" path — no LLM call needed. Returns None if
        the session memory is too sparse to be useful.
        """
        if not self._worklog and self._task_description == "(not yet identified)":
            return None

        return _TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            current_state=self._current_state,
            task=self._task_description,
            files=self._format_list(sorted(self._files_touched)) or "(none)",
            workflow=self._format_list(self._workflow_summary()) or "(no progress yet)",
            errors=self._format_list(self._errors[-10:]) or "(none)",
            learnings=self._format_list(self._learnings[-10:]) or "(none)",
            worklog=self._format_list(self._worklog[-30:]) or "(empty)",
        )

    def _workflow_summary(self, limit: int = 10) -> list[str]:
        """Deduplicated recent commands for workflow progress section."""
        seen: set[str] = set()
        result: list[str] = []
        for entry in reversed(self._worklog):
            clean = entry.replace(" **FAILED**", "")
            if clean not in seen:
                seen.add(clean)
                result.append(clean)
            if len(result) >= limit:
                break
        result.reverse()
        return result

    @staticmethod
    def _format_list(items: list[str] | set[str]) -> str:
        """Format a list of items as markdown bullets."""
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)
