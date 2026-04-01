"""Context management — tool result storage, compaction, and system prompt injection."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .schema import Message


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

def compute_compact_threshold(
    context_window: int = 200_000,
    compact_threshold_pct: float = 0.85,
    compaction_reserve: int = 20_000,
) -> int:
    """Compute the absolute token threshold for compaction.

    Args:
        context_window: Total context window size in tokens (e.g. 200_000).
        compact_threshold_pct: Fraction of context window at which to trigger
            compaction. Should be 0.85–0.90; 0.95 triggers too late.
        compaction_reserve: Tokens reserved for the compaction summary output.

    Returns:
        Effective token threshold (context_window * pct - reserve).
    """
    return int(context_window * compact_threshold_pct) - compaction_reserve


async def compact_messages(
    messages: list[Message],
    llm_client: Any,
    token_threshold: int | None = None,
    keep_recent: int = 6,
    *,
    context_window: int = 200_000,
    compact_threshold_pct: float = 0.85,
    compaction_reserve: int = 20_000,
) -> list[Message]:
    """Summarize older turns when estimated tokens exceed threshold.

    The threshold is percentage-based by default: 85% of the context window
    minus a reserve for the compaction summary itself. An explicit
    ``token_threshold`` overrides the percentage calculation.

    Keeps the system message, a summary of old turns, and the most recent turns.
    Returns a new message list (does not mutate the original).
    """
    if token_threshold is None:
        token_threshold = compute_compact_threshold(
            context_window, compact_threshold_pct, compaction_reserve
        )

    current_tokens = estimate_tokens(messages)
    if current_tokens < token_threshold:
        return messages

    # Separate system message from conversation
    system_msg = messages[0] if messages and messages[0].role == "system" else None
    conversation = messages[1:] if system_msg else messages

    if len(conversation) <= keep_recent:
        return messages  # Not enough to compact

    old_turns = conversation[:-keep_recent]
    recent_turns = conversation[-keep_recent:]

    # Build summary prompt
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
