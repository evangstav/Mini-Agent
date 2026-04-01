"""autoDream — background cross-session memory consolidation.

After a session ends, reviews the conversation and updates persistent memory
files. Four phases:

1. Orient   — read existing memories from the memory directory
2. Gather   — scan the conversation transcript for memorable signals
3. Consolidate — write or update memory files
4. Prune    — clean up the memory index (MEMORY.md)

The dream runs as a background subagent with read-only bash access and
write access limited to the memory directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .hooks import HookEvent, HookRegistry, SessionEndPayload
from .schema import Message

logger = logging.getLogger(__name__)


# Default location for persistent memories
DEFAULT_MEMORY_DIR = ".claude/memory"
MEMORY_INDEX = "MEMORY.md"

# LLM prompt for the consolidation phase
DREAM_SYSTEM_PROMPT = """\
You are a memory consolidation agent. Your job is to review a completed \
conversation and extract durable, cross-session knowledge worth remembering.

You will receive:
1. Existing memories (may be empty for a fresh project)
2. The conversation transcript

Your task — produce a JSON object with one key:
- "updates": list of memory file operations (create, update, or delete)

Each item in "updates" is an object:
  {"action": "create"|"update"|"delete",
   "filename": "some_memory.md",
   "content": "full file content including frontmatter (for create/update)"}

The MEMORY.md index is rebuilt automatically from the memory files — do not \
include index entries in your response.

Rules:
- Only save information that will be useful in FUTURE conversations.
- Do NOT save: code patterns derivable from reading the code, git history, \
  debugging solutions (the fix is in the code), ephemeral task details.
- DO save: user preferences, feedback/corrections, project context, external \
  references, role information.
- Use frontmatter format: ---\\nname: ...\\ndescription: ...\\ntype: user|feedback|project|reference\\n---
- Keep descriptions under 100 chars — they're used for relevance matching.
- If existing memories cover the same topic, update them rather than creating duplicates.
- If a memory is now stale (contradicted by conversation), delete it.
- If nothing is worth remembering, return {"updates": []}.
- Return ONLY valid JSON, no markdown fences or commentary.
"""


class DreamConsolidator:
    """Runs the four-phase memory consolidation after a session ends.

    Usage:
        consolidator = DreamConsolidator(memory_dir="/path/to/memories")
        hook_registry.on(HookEvent.SESSION_END, consolidator.on_session_end)
    """

    def __init__(
        self,
        memory_dir: str | Path = DEFAULT_MEMORY_DIR,
        llm_client: Any | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.llm_client = llm_client
        self._lock = asyncio.Lock()

    def register(self, registry: HookRegistry) -> None:
        """Register this consolidator with a hook registry."""
        registry.on(HookEvent.SESSION_END, self.on_session_end)

    async def on_session_end(self, payload: SessionEndPayload) -> None:
        """Hook callback — runs the full dream cycle.

        Uses a lock to prevent concurrent memory file writes when
        multiple session-end events fire simultaneously.
        """
        if self.llm_client is None:
            return
        async with self._lock:
            await self.run(payload.messages, payload.metadata)

    async def run(
        self, messages: list[Message], metadata: dict[str, Any] | None = None
    ) -> DreamResult:
        """Execute the four-phase dream consolidation.

        Returns a DreamResult describing what was done.
        """
        # Phase 1: Orient — read existing memories
        existing = self._orient()

        # Phase 2: Gather — build transcript for LLM
        transcript = self._gather(messages)

        # Phase 3: Consolidate — ask LLM what to remember
        operations = await self._consolidate(existing, transcript)

        # Phase 4: Prune — apply operations and rebuild index
        applied = self._prune(operations)

        return applied

    # ── Phase 1: Orient ──────────────────────────────────────────────

    def _orient(self) -> dict[str, str]:
        """Read all existing memory files and the index.

        Returns a dict of filename → content for all .md files in memory_dir.
        """
        memories: dict[str, str] = {}
        if not self.memory_dir.exists():
            return memories

        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == MEMORY_INDEX:
                memories["MEMORY.md"] = path.read_text(encoding="utf-8")
            else:
                memories[path.name] = path.read_text(encoding="utf-8")
        return memories

    # ── Phase 2: Gather ──────────────────────────────────────────────

    def _gather(self, messages: list[Message], max_chars: int = 120_000) -> str:
        """Build a compact transcript from the conversation messages.

        Keeps the most recent messages first (they're most relevant),
        truncating older ones if the transcript exceeds max_chars.
        """
        parts: list[str] = []
        char_count = 0

        # Walk from most recent to oldest
        for msg in reversed(messages):
            if msg.role == "system":
                continue  # Skip system prompt (it's not conversation)
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            # Truncate individual messages that are very long
            if len(content) > 2000:
                content = content[:2000] + "..."
            line = f"[{msg.role}]: {content}"
            if char_count + len(line) > max_chars:
                break
            parts.append(line)
            char_count += len(line)

        # Reverse back to chronological order
        parts.reverse()
        return "\n".join(parts)

    # ── Phase 3: Consolidate ─────────────────────────────────────────

    async def _consolidate(
        self, existing: dict[str, str], transcript: str
    ) -> list[dict[str, Any]]:
        """Ask the LLM to decide what memories to create/update/delete.

        Returns a list of operation dicts from the LLM response.
        """
        if self.llm_client is None:
            return []

        # Build the user prompt
        existing_block = ""
        if existing:
            existing_parts = []
            for fname, content in existing.items():
                existing_parts.append(f"### {fname}\n{content}")
            existing_block = "\n\n".join(existing_parts)
        else:
            existing_block = "(no existing memories)"

        user_prompt = (
            "## Existing Memories\n\n"
            f"{existing_block}\n\n"
            "## Conversation Transcript\n\n"
            f"{transcript}\n\n"
            "Produce a JSON object with 'updates' as described."
        )

        try:
            response = await self.llm_client.generate(
                messages=[
                    Message(role="system", content=DREAM_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ]
            )
            result = json.loads(response.content)
            return result.get("updates", [])
        except (json.JSONDecodeError, Exception):
            return []

    # ── Phase 4: Prune ───────────────────────────────────────────────

    def _prune(self, operations: list[dict[str, Any]]) -> DreamResult:
        """Apply memory operations and rebuild the index.

        Returns a DreamResult summarizing what was done.
        """
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []

        if not operations:
            return DreamResult(created=created, updated=updated, deleted=deleted)

        # Ensure memory directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        for op in operations:
            action = op.get("action", "")
            filename = op.get("filename", "")
            content = op.get("content", "")

            if not filename or filename == MEMORY_INDEX:
                continue  # Skip invalid or index-only ops

            # Sanitize filename — prevent path traversal
            safe_name = Path(filename).name
            if not safe_name.endswith(".md"):
                safe_name += ".md"
            target = self.memory_dir / safe_name

            if action == "create":
                target.write_text(content, encoding="utf-8")
                created.append(safe_name)
            elif action == "update":
                target.write_text(content, encoding="utf-8")
                updated.append(safe_name)
            elif action == "delete":
                if target.exists():
                    target.unlink()
                deleted.append(safe_name)

        # Rebuild MEMORY.md index from surviving files
        self._rebuild_index()

        return DreamResult(created=created, updated=updated, deleted=deleted)

    def _rebuild_index(self) -> None:
        """Rebuild MEMORY.md from all memory files in the directory.

        Reads each .md file's frontmatter to extract name and description,
        then writes a clean index.
        """
        if not self.memory_dir.exists():
            return

        lines: list[str] = ["# Memory Index", ""]
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == MEMORY_INDEX:
                continue
            # Extract name and description from frontmatter
            name, description = _parse_frontmatter(path)
            if name:
                lines.append(f"- [{name}]({path.name}) — {description}")
            else:
                lines.append(f"- [{path.stem}]({path.name})")

        index_path = self.memory_dir / MEMORY_INDEX
        if len(lines) > 2:  # Has entries beyond header
            index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif index_path.exists():
            index_path.unlink()  # Remove empty index


class DreamResult:
    """Summary of what the dream consolidation did."""

    def __init__(
        self,
        created: list[str] | None = None,
        updated: list[str] | None = None,
        deleted: list[str] | None = None,
    ) -> None:
        self.created = created or []
        self.updated = updated or []
        self.deleted = deleted or []

    @property
    def total_changes(self) -> int:
        return len(self.created) + len(self.updated) + len(self.deleted)

    def __repr__(self) -> str:
        return (
            f"DreamResult(created={self.created}, "
            f"updated={self.updated}, deleted={self.deleted})"
        )


def _parse_frontmatter(path: Path) -> tuple[str, str]:
    """Extract name and description from YAML-ish frontmatter.

    Returns (name, description) or ("", "") if no frontmatter found.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", ""

    if not text.startswith("---"):
        return "", ""

    end = text.find("---", 3)
    if end == -1:
        return "", ""

    frontmatter = text[3:end]
    name = ""
    description = ""
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[5:].strip()
        elif line.startswith("description:"):
            description = line[12:].strip()
    return name, description
