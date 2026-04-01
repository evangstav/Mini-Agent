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
    """Rough token estimate: ~3 chars per token (conservative for code-heavy context)."""
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
    return total_chars // 3


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
        role="assistant",
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

        memories = self._load_memories()
        if memories:
            parts.append(f"\n\n# Persistent Memory\n\n{memories}")

        git_info = self._get_git_info()
        if git_info:
            parts.append(f"\n\n# Git Context\n\n{git_info}")

        return "\n".join(parts)

    def _load_memories(self) -> str | None:
        """Load persistent memory files from the memory directory.

        Reads MEMORY.md index and referenced memory files from
        .claude/memory/ in the project directory.

        Returns merged memory content or None if no memories found.
        """
        memory_dir = Path(self.project_dir) / ".claude" / "memory"
        if not memory_dir.is_dir():
            return None

        index_path = memory_dir / "MEMORY.md"
        if not index_path.is_file():
            return None

        sections: list[str] = []
        char_budget = 4000
        chars_used = 0

        # Include the index
        index_content = index_path.read_text(encoding="utf-8")
        sections.append(f"## Memory Index\n\n{index_content}")
        chars_used += len(index_content)

        # Include individual memory files
        for path in sorted(memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            remaining = char_budget - chars_used
            if remaining <= 0:
                break
            content = path.read_text(encoding="utf-8")
            if len(content) > remaining:
                content = content[:remaining] + "\n[... truncated]"
            sections.append(f"## {path.stem}\n\n{content}")
            chars_used += len(content)

        return "\n\n".join(sections) if len(sections) > 1 else None

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


# --- Handoff Context Extraction ---


async def extract_handoff_context(
    messages: list[Message],
    new_goal: str,
    llm_client: Any,
    system_prompt: str | None = None,
    max_context_chars: int = 12_000,
) -> list[Message]:
    """Extract relevant context from a prior conversation for a new task.

    Instead of carrying forward the entire conversation (context pollution)
    or summarizing everything (lossy), this uses an LLM to selectively extract
    only the context relevant to the new goal.

    Args:
        messages: Prior conversation history.
        new_goal: Description of the new task/goal.
        llm_client: LLM client for extraction (can be a cheaper/faster model).
        system_prompt: Optional system prompt for the new thread. If None,
            the system prompt from the prior conversation is reused.
        max_context_chars: Budget for extracted context (default 12K chars ~3K tokens).

    Returns:
        Fresh message list: [system_prompt, extracted_context (user), new_goal (user)].
        If extraction fails, returns [system_prompt, new_goal] with no prior context.
    """
    # Preserve original system prompt
    original_system = None
    conversation: list[Message] = []
    for msg in messages:
        if msg.role == "system" and original_system is None:
            original_system = msg
        else:
            conversation.append(msg)

    effective_system = (
        Message(role="system", content=system_prompt)
        if system_prompt
        else original_system
        or Message(role="system", content="You are a helpful assistant.")
    )

    # If no conversation history, just return system + goal
    if not conversation:
        return [effective_system, Message(role="user", content=new_goal)]

    # Build a digest of the prior conversation for the extractor
    digest_parts: list[str] = []
    char_budget = max_context_chars * 3  # Allow extractor to see ~3x what it outputs
    chars_used = 0
    for msg in conversation:
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        # Truncate individual messages to avoid one huge tool result dominating
        if len(content) > 2000:
            content = content[:2000] + "..."
        line = f"[{msg.role}]: {content}"
        if chars_used + len(line) > char_budget:
            digest_parts.append("[... earlier messages omitted for brevity ...]")
            break
        digest_parts.append(line)
        chars_used += len(line)

    conversation_digest = "\n".join(digest_parts)

    extraction_prompt = (
        "You are a context extraction assistant. Given a prior conversation and a new goal, "
        "extract ONLY the information from the prior conversation that is relevant to the new goal.\n\n"
        "Rules:\n"
        "- Include: file paths, code snippets, decisions, constraints, error messages, "
        "and architectural context that the new task needs.\n"
        "- Exclude: completed work unrelated to the new goal, debugging dead-ends, "
        "social pleasantries, and tool outputs that aren't referenced.\n"
        "- Be concise. Use bullet points. Preserve exact values (paths, names, numbers).\n"
        "- If nothing from the prior conversation is relevant, respond with exactly: "
        '"NO_RELEVANT_CONTEXT"\n\n'
        f"## New Goal\n{new_goal}\n\n"
        f"## Prior Conversation\n{conversation_digest}\n\n"
        "## Extracted Context\n"
        "Extract only what's relevant to the new goal:"
    )

    try:
        response = await llm_client.generate(
            messages=[
                Message(role="system", content="You extract relevant context from conversations. Be precise and concise."),
                Message(role="user", content=extraction_prompt),
            ]
        )
        extracted = response.content.strip()
    except Exception:
        # Extraction failed — start fresh with just the goal
        return [effective_system, Message(role="user", content=new_goal)]

    # Build the fresh thread
    result = [effective_system]

    if extracted and extracted != "NO_RELEVANT_CONTEXT":
        # Enforce max_context_chars on the extraction output
        if len(extracted) > max_context_chars:
            extracted = extracted[:max_context_chars] + "\n[... context truncated]"
        result.append(
            Message(
                role="user",
                content=f"[Relevant context from prior conversation]:\n{extracted}",
            )
        )

    result.append(Message(role="user", content=new_goal))
    return result
