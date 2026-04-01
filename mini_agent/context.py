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


# --- Tool Result Pruning ---

def prune_tool_results(
    messages: list[Message],
    protect_chars: int = 160_000,
    prune_threshold: int = 80_000,
) -> list[Message]:
    """Remove bloated tool results outside the recent protection window.

    Scans backward from the most recent message. Tool-result messages within the
    first *protect_chars* characters of cumulative content are kept intact.
    Beyond that window, any tool-result message whose content exceeds
    *prune_threshold* characters is replaced with a short placeholder.

    This is cheaper than compaction — no LLM call, just mechanical pruning.
    Should run before compact_messages().

    Default protect_chars ~160K chars ≈ 40K tokens.
    Default prune_threshold ~80K chars ≈ 20K tokens.
    """
    if not messages:
        return messages

    # Walk backward, accumulating character counts to find the protection boundary
    cumulative = 0
    # Index → True means "outside protection window AND eligible for pruning"
    prune_indices: set[int] = set()

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_chars = len(msg.content) if isinstance(msg.content, str) else len(json.dumps(msg.content))

        if cumulative >= protect_chars:
            # Outside protection window — check if this is a large tool result
            if msg.role == "tool" and msg_chars > prune_threshold:
                prune_indices.add(i)

        cumulative += msg_chars

    if not prune_indices:
        return messages  # Nothing to prune

    # Build new list, replacing pruned messages
    result = []
    for i, msg in enumerate(messages):
        if i in prune_indices:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            result.append(Message(
                role=msg.role,
                content=f"[Tool result pruned — was {len(content)} chars. Re-run the tool if needed.]",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            result.append(msg)
    return result


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

        instructions = self._discover_claude_instructions()
        if instructions:
            parts.append(f"\n\n# Project Instructions\n\n{instructions}")

        git_info = self._get_git_info()
        if git_info:
            parts.append(f"\n\n# Git Context\n\n{git_info}")

        return "\n".join(parts)

    def _discover_claude_instructions(self) -> str | None:
        """Discover and merge hierarchical CLAUDE.md files.

        Merges instruction files from three levels (each level adds context):
          1. Global:    ~/.claude/CLAUDE.md
          2. Project:   CLAUDE.md files walking up from project_dir to git/fs root
          3. Local:     .claude/rules/*.md in project_dir

        Returns merged content or None if no files found.
        """
        sections: list[str] = []
        char_budget = 8000  # Total budget across all levels
        chars_used = 0

        # Level 1: Global (~/.claude/CLAUDE.md)
        global_path = Path.home() / ".claude" / "CLAUDE.md"
        chars_used = self._append_file(
            global_path, "Global Instructions", sections, char_budget, chars_used
        )

        # Level 2: Project CLAUDE.md files (walk up from project_dir to root)
        for claude_path in self._walk_up_claude_md():
            chars_used = self._append_file(
                claude_path,
                f"Project Instructions ({claude_path.parent.name}/)",
                sections,
                char_budget,
                chars_used,
            )

        # Level 3: .claude/rules/*.md in project_dir
        rules_dir = Path(self.project_dir) / ".claude" / "rules"
        if rules_dir.is_dir():
            for rule_file in sorted(rules_dir.glob("*.md")):
                if rule_file.is_file():
                    chars_used = self._append_file(
                        rule_file,
                        f"Rule: {rule_file.stem}",
                        sections,
                        char_budget,
                        chars_used,
                    )

        return "\n\n---\n\n".join(sections) if sections else None

    def _walk_up_claude_md(self) -> list[Path]:
        """Walk up from project_dir, collecting CLAUDE.md files until git/fs root.

        Returns paths ordered from outermost (root) to innermost (project_dir)
        so that more specific instructions appear last and can override.
        """
        found: list[Path] = []
        current = Path(self.project_dir).resolve()

        while True:
            candidate = current / "CLAUDE.md"
            if candidate.is_file():
                found.append(candidate)

            # Stop at git root or filesystem root
            if (current / ".git").exists() or current == current.parent:
                break
            current = current.parent

        # Reverse: outermost first, project_dir last (most specific wins)
        found.reverse()
        return found

    def _append_file(
        self,
        path: Path,
        label: str,
        sections: list[str],
        char_budget: int,
        chars_used: int,
    ) -> int:
        """Read a file and append it as a labeled section. Returns updated chars_used."""
        if not path.is_file():
            return chars_used

        remaining = char_budget - chars_used
        if remaining <= 0:
            return chars_used

        content = path.read_text(encoding="utf-8")
        if len(content) > remaining:
            content = content[:remaining] + f"\n\n[... {label} truncated]"

        sections.append(f"## {label}\n\n{content}")
        return chars_used + len(content)

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
