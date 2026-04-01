"""Context management — tool result storage, compaction, session memory, and system prompt injection."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import Message


# --- Session Memory ---


class SessionMemory:
    """Running markdown notes maintained during a conversation.

    Provides a free compaction summary — no LLM call needed because the
    memory is already a human-readable summary of key decisions and facts.
    Persists to disk so it survives crashes.
    """

    def __init__(self, storage_path: str = ".runtime/session_memory.md"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Load existing session memory from disk."""
        if not self.path.exists():
            return
        try:
            content = self.path.read_text(encoding="utf-8")
            for line in content.strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    # Parse "- [category] note" or "- note"
                    text = line[2:]
                    if text.startswith("[") and "]" in text:
                        bracket_end = text.index("]")
                        category = text[1:bracket_end]
                        note = text[bracket_end + 1:].strip()
                    else:
                        category = "general"
                        note = text
                    self._entries.append({"category": category, "note": note})
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        """Persist session memory to disk."""
        lines = []
        for entry in self._entries:
            lines.append(f"- [{entry['category']}] {entry['note']}")
        self.path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def add(self, note: str, category: str = "general") -> None:
        """Add a note to session memory."""
        self._entries.append({"category": category, "note": note})
        self._save()

    def get_summary(self) -> str:
        """Return session memory as a markdown summary for compaction."""
        if not self._entries:
            return ""
        lines = ["# Session Memory"]
        by_category: dict[str, list[str]] = {}
        for entry in self._entries:
            by_category.setdefault(entry["category"], []).append(entry["note"])
        for cat, notes in by_category.items():
            lines.append(f"\n## {cat}")
            for note in notes:
                lines.append(f"- {note}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all session memory entries."""
        self._entries = []
        self._save()

    @property
    def entries(self) -> list[dict[str, str]]:
        """Read-only access to entries."""
        return list(self._entries)


# --- Microcompaction ---


def microcompact_messages(
    messages: list[Message],
    keep_recent: int = 4,
    max_tool_content: int = 200,
) -> list[Message]:
    """Replace stale tool results with short placeholders — no API calls needed.

    This is the cheapest form of context relief. Old tool result messages have
    their content replaced with a brief reference, preserving the conversation
    structure while dramatically reducing token count.

    Args:
        messages: Full message list.
        keep_recent: Number of recent messages to leave untouched.
        max_tool_content: Max chars to keep for compacted tool results.

    Returns:
        New message list with old tool results compacted.
    """
    if len(messages) <= keep_recent + 1:  # +1 for system message
        return messages

    # Split: system | old | recent
    system_msg = messages[0] if messages and messages[0].role == "system" else None
    conversation = messages[1:] if system_msg else messages

    if len(conversation) <= keep_recent:
        return messages

    boundary = len(conversation) - keep_recent
    old_msgs = conversation[:boundary]
    recent_msgs = conversation[boundary:]

    compacted_old = []
    for msg in old_msgs:
        if msg.role == "tool":
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if len(content) > max_tool_content:
                preview = content[:max_tool_content].rstrip()
                tool_name = msg.name or "unknown"
                compacted_content = (
                    f"[{tool_name} result — {len(content)} chars, compacted]\n{preview}..."
                )
                compacted_old.append(Message(
                    role=msg.role,
                    content=compacted_content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                ))
            else:
                compacted_old.append(msg)
        elif msg.role == "assistant" and msg.thinking:
            # Strip old thinking blocks — they're useful live but not worth keeping
            compacted_old.append(Message(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
            ))
        else:
            compacted_old.append(msg)

    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(compacted_old)
    result.extend(recent_msgs)
    return result


# --- Tool Result Storage ---

class ToolResultStore:
    """Persists large tool results to disk, keeps previews in context."""

    def __init__(self, storage_dir: str = ".runtime/tool_results", preview_chars: int = 1500):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.preview_chars = preview_chars

    def store_if_large(self, content: str, tool_call_id: str) -> str:
        """If content exceeds preview_chars, store to disk and return a preview.

        Returns the (possibly truncated) content to keep in context.
        """
        if len(content) <= self.preview_chars:
            return content

        # Persist full result
        file_hash = hashlib.sha256(tool_call_id.encode()).hexdigest()[:12]
        result_path = self.storage_dir / f"{file_hash}.txt"
        result_path.write_text(content, encoding="utf-8")

        # Return preview with reference
        preview = content[: self.preview_chars]
        total_chars = len(content)
        return (
            f"{preview}\n\n"
            f"[... truncated — {total_chars} chars total, "
            f"full result stored at {result_path}]"
        )

    def retrieve(self, tool_call_id: str) -> str | None:
        """Retrieve full tool result from disk."""
        file_hash = hashlib.sha256(tool_call_id.encode()).hexdigest()[:12]
        result_path = self.storage_dir / f"{file_hash}.txt"
        if result_path.exists():
            return result_path.read_text(encoding="utf-8")
        return None


# --- Token Estimation ---

def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        elif isinstance(msg.content, list):
            total_chars += len(json.dumps(msg.content))
        if msg.thinking:
            total_chars += len(msg.thinking)
        if msg.tool_calls:
            total_chars += len(json.dumps([tc.model_dump() for tc in msg.tool_calls]))
    return total_chars // 4


# --- Context Compaction ---

async def compact_messages(
    messages: list[Message],
    llm_client: Any,
    token_threshold: int = 80_000,
    keep_recent: int = 6,
    session_memory: SessionMemory | None = None,
) -> list[Message]:
    """Compact context using a tiered strategy when tokens exceed threshold.

    Tier 1 (free): Microcompaction — strip stale tool results and thinking blocks.
    Tier 2 (free): Session memory — use running notes as summary if available.
    Tier 3 (costly): LLM summarization — call the model to summarize old turns.

    Keeps the system message, a summary of old turns, and the most recent turns.
    Returns a new message list (does not mutate the original).
    """
    current_tokens = estimate_tokens(messages)
    if current_tokens < token_threshold:
        return messages

    # Tier 1: Microcompaction (always try first — it's free)
    compacted = microcompact_messages(messages, keep_recent=keep_recent)
    if estimate_tokens(compacted) < token_threshold:
        return compacted

    # Still over threshold — need to summarize and drop old turns
    system_msg = compacted[0] if compacted and compacted[0].role == "system" else None
    conversation = compacted[1:] if system_msg else compacted

    if len(conversation) <= keep_recent:
        return compacted  # Not enough to compact further

    old_turns = conversation[:-keep_recent]
    recent_turns = conversation[-keep_recent:]

    # Tier 2: Use session memory as free summary if available
    if session_memory:
        memory_summary = session_memory.get_summary()
        if memory_summary:
            summary_msg = Message(
                role="user",
                content=(
                    f"[Context summary of {len(old_turns)} earlier messages — "
                    f"from session memory]:\n{memory_summary}"
                ),
            )
            result = []
            if system_msg:
                result.append(system_msg)
            result.append(summary_msg)
            result.extend(recent_turns)
            return result

    # Tier 3: LLM summarization (most expensive)
    old_text_parts = []
    for msg in old_turns:
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        old_text_parts.append(f"[{msg.role}]: {content[:500]}")

    summary_prompt = (
        "Summarize the following conversation turns into a concise paragraph. "
        "Preserve key facts, decisions, tool results, and any errors encountered. "
        "Be brief but complete.\n\n" + "\n".join(old_text_parts)
    )

    try:
        summary_response = await llm_client.generate(
            messages=[
                Message(role="system", content="You are a conversation summarizer."),
                Message(role="user", content=summary_prompt),
            ]
        )
        summary_text = summary_response.content
    except Exception:
        # If summarization fails, fall back to naive truncation
        summary_text = f"[Prior context: {len(old_turns)} messages omitted due to context limits]"

    summary_msg = Message(
        role="user",
        content=f"[Context summary of {len(old_turns)} earlier messages]:\n{summary_text}",
    )

    result = []
    if system_msg:
        result.append(system_msg)
    result.append(summary_msg)
    result.extend(recent_turns)
    return result


# --- System Prompt Builder ---

class SystemPromptBuilder:
    """Builds system prompt from base prompt, CLAUDE.md, and git info."""

    def __init__(self, base_prompt: str, project_dir: str | None = None):
        self.base_prompt = base_prompt
        self.project_dir = project_dir or os.getcwd()

    def build(self) -> str:
        """Assemble the full system prompt."""
        parts = [self.base_prompt]

        claude_md = self._read_claude_md()
        if claude_md:
            parts.append(f"\n\n# Project Instructions (CLAUDE.md)\n\n{claude_md}")

        git_info = self._get_git_info()
        if git_info:
            parts.append(f"\n\n# Git Context\n\n{git_info}")

        return "\n".join(parts)

    def _read_claude_md(self) -> str | None:
        """Read CLAUDE.md from project directory if it exists."""
        claude_path = Path(self.project_dir) / "CLAUDE.md"
        if claude_path.exists():
            content = claude_path.read_text(encoding="utf-8")
            # Limit to avoid bloating system prompt
            if len(content) > 4000:
                return content[:4000] + "\n\n[... CLAUDE.md truncated]"
            return content
        return None

    def _get_git_info(self) -> str | None:
        """Get current branch and recent commits."""
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                timeout=5,
            )
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                timeout=5,
            )
            if branch.returncode != 0:
                return None

            parts = [f"Branch: {branch.stdout.strip()}"]
            if log.returncode == 0 and log.stdout.strip():
                parts.append(f"Recent commits:\n{log.stdout.strip()}")
            return "\n".join(parts)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
