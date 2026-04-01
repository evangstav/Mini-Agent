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

async def compact_messages(
    messages: list[Message],
    llm_client: Any,
    token_threshold: int = 80_000,
    keep_recent: int = 6,
) -> list[Message]:
    """Summarize older turns when estimated tokens exceed threshold.

    Keeps the system message, a summary of old turns, and the most recent turns.
    Returns a new message list (does not mutate the original).
    """
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

    # Build structured summary prompt
    old_text_parts = []
    for msg in old_turns:
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        old_text_parts.append(f"[{msg.role}]: {content[:500]}")

    conversation_block = "\n".join(old_text_parts)

    summary_prompt = (
        "Summarize the following conversation into a structured compaction summary. "
        "Use EXACTLY these sections (omit a section only if there is genuinely "
        "nothing for it):\n\n"
        "## Completed Work\n"
        "What was accomplished: files created/modified, bugs fixed, features added, "
        "commands run and their outcomes.\n\n"
        "## Current File State\n"
        "Which files were touched and their current status (created, modified, deleted). "
        "Include key structural decisions (e.g. 'moved X from module A to B').\n\n"
        "## Active Tasks\n"
        "Work still in progress or pending. Include any failing tests or unresolved errors.\n\n"
        "## Next Steps\n"
        "What should happen next, in priority order.\n\n"
        "## Key Decisions & Constraints\n"
        "Important decisions made and why, user preferences or constraints that must "
        "be preserved, and any discovered blockers.\n\n"
        "Be concise but preserve all actionable information. "
        "Prefer bullet points over prose.\n\n"
        "---\n\n" + conversation_block
    )

    try:
        summary_response = await llm_client.generate(
            messages=[
                Message(role="system", content="You produce structured compaction summaries of coding agent conversations. Be precise and preserve actionable detail."),
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
