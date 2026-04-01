"""Test cases for the dream (autoDream) module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mini_agent.dream import (
    DEFAULT_MEMORY_DIR,
    MEMORY_INDEX,
    DreamConsolidator,
    DreamResult,
    _parse_frontmatter,
)
from mini_agent.hooks import HookEvent, HookRegistry, SessionEndPayload
from mini_agent.schema import LLMResponse, Message


@pytest.fixture
def memory_dir(tmp_path):
    """Temporary memory directory."""
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def mock_llm():
    return AsyncMock()


# ── Phase 1: Orient ─────────────────────────────────────────────


def test_orient_empty_dir(memory_dir):
    """Orient returns empty dict when memory dir is empty."""
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._orient()
    assert result == {}


def test_orient_reads_memory_files(memory_dir):
    """Orient reads all .md files in memory dir."""
    (memory_dir / "user_role.md").write_text(
        "---\nname: Role\ndescription: user role\ntype: user\n---\nSenior dev"
    )
    (memory_dir / "feedback_testing.md").write_text(
        "---\nname: Testing\ndescription: test prefs\ntype: feedback\n---\nUse real DB"
    )

    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._orient()

    assert "user_role.md" in result
    assert "Senior dev" in result["user_role.md"]
    assert "feedback_testing.md" in result
    assert len(result) == 2


def test_orient_includes_index(memory_dir):
    """Orient reads MEMORY.md if present."""
    (memory_dir / MEMORY_INDEX).write_text("# Memory Index\n- [Role](user_role.md)")
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._orient()
    assert "MEMORY.md" in result


def test_orient_nonexistent_dir(tmp_path):
    """Orient returns empty dict when dir doesn't exist."""
    c = DreamConsolidator(memory_dir=tmp_path / "nonexistent")
    result = c._orient()
    assert result == {}


# ── Phase 2: Gather ──────────────────────────────────────────────


def test_gather_basic_transcript(memory_dir):
    """Gather builds a readable transcript."""
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    transcript = c._gather(messages)

    assert "[user]: Hello" in transcript
    assert "[assistant]: Hi there!" in transcript
    assert "[system]" not in transcript  # System messages are skipped


def test_gather_truncates_long_messages(memory_dir):
    """Gather truncates individual messages over 2000 chars."""
    messages = [
        Message(role="user", content="x" * 5000),
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    transcript = c._gather(messages)

    assert len(transcript) < 5000
    assert "..." in transcript


def test_gather_respects_max_chars(memory_dir):
    """Gather stops adding messages when max_chars is reached."""
    messages = [
        Message(role="user", content="old message " * 100),
        Message(role="assistant", content="old reply " * 100),
        Message(role="user", content="recent message"),
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    transcript = c._gather(messages, max_chars=200)

    # Should include the most recent messages first (reversed walk)
    assert "recent message" in transcript


def test_gather_empty_messages(memory_dir):
    """Gather handles empty message list."""
    c = DreamConsolidator(memory_dir=memory_dir)
    transcript = c._gather([])
    assert transcript == ""


# ── Phase 3: Consolidate ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_creates_memories(memory_dir, mock_llm):
    """Consolidate calls LLM and returns operations."""
    mock_llm.generate.return_value = LLMResponse(
        content=json.dumps({
            "updates": [
                {
                    "action": "create",
                    "filename": "user_role.md",
                    "content": "---\nname: Role\ndescription: user role\ntype: user\n---\nDev",
                }
            ],
            "index_lines": ["- [Role](user_role.md) — user role"],
        }),
        finish_reason="stop",
    )

    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    ops = await c._consolidate({}, "transcript text")

    assert len(ops) == 1
    assert ops[0]["action"] == "create"
    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_consolidate_returns_empty_on_llm_error(memory_dir, mock_llm):
    """Consolidate returns empty list if LLM fails."""
    mock_llm.generate.side_effect = Exception("API error")
    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    ops = await c._consolidate({}, "transcript")
    assert ops == []


@pytest.mark.asyncio
async def test_consolidate_returns_empty_on_invalid_json(memory_dir, mock_llm):
    """Consolidate returns empty list if LLM returns invalid JSON."""
    mock_llm.generate.return_value = LLMResponse(
        content="not json", finish_reason="stop"
    )
    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    ops = await c._consolidate({}, "transcript")
    assert ops == []


@pytest.mark.asyncio
async def test_consolidate_no_llm(memory_dir):
    """Consolidate returns empty list when no LLM is configured."""
    c = DreamConsolidator(memory_dir=memory_dir, llm_client=None)
    ops = await c._consolidate({}, "transcript")
    assert ops == []


# ── Phase 4: Prune ───────────────────────────────────────────────


def test_prune_creates_files(memory_dir):
    """Prune creates memory files and rebuilds index."""
    ops = [
        {
            "action": "create",
            "filename": "user_role.md",
            "content": "---\nname: Role\ndescription: user role\ntype: user\n---\nDev",
        }
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.created == ["user_role.md"]
    assert (memory_dir / "user_role.md").exists()
    assert (memory_dir / MEMORY_INDEX).exists()


def test_prune_updates_files(memory_dir):
    """Prune updates existing memory files."""
    (memory_dir / "user_role.md").write_text("old content")

    ops = [
        {
            "action": "update",
            "filename": "user_role.md",
            "content": "---\nname: Role\ndescription: updated role\ntype: user\n---\nSenior Dev",
        }
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.updated == ["user_role.md"]
    assert "Senior Dev" in (memory_dir / "user_role.md").read_text()


def test_prune_deletes_files(memory_dir):
    """Prune deletes stale memory files."""
    (memory_dir / "old_memory.md").write_text("stale")

    ops = [{"action": "delete", "filename": "old_memory.md"}]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.deleted == ["old_memory.md"]
    assert not (memory_dir / "old_memory.md").exists()


def test_prune_prevents_path_traversal(memory_dir):
    """Prune sanitizes filenames to prevent path traversal."""
    ops = [
        {
            "action": "create",
            "filename": "../../../etc/passwd",
            "content": "evil",
        }
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    # Should create "passwd.md" in memory_dir, not traverse
    assert result.created == ["passwd.md"]
    assert not Path("/etc/passwd").read_text().startswith("evil")


def test_prune_skips_index_filename(memory_dir):
    """Prune ignores operations targeting MEMORY.md directly."""
    ops = [
        {"action": "create", "filename": MEMORY_INDEX, "content": "hacked"}
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.total_changes == 0


def test_prune_empty_operations(memory_dir):
    """Prune with empty operations returns empty result."""
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune([])

    assert result.total_changes == 0


def test_prune_delete_nonexistent(memory_dir):
    """Prune handles deleting a file that doesn't exist."""
    ops = [{"action": "delete", "filename": "nonexistent.md"}]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.deleted == ["nonexistent.md"]


def test_prune_adds_md_extension(memory_dir):
    """Prune adds .md extension to filenames that lack it."""
    ops = [
        {
            "action": "create",
            "filename": "user_role",
            "content": "---\nname: Role\ndescription: desc\ntype: user\n---\ncontent",
        }
    ]
    c = DreamConsolidator(memory_dir=memory_dir)
    result = c._prune(ops)

    assert result.created == ["user_role.md"]
    assert (memory_dir / "user_role.md").exists()


# ── Index Rebuild ────────────────────────────────────────────────


def test_rebuild_index_from_files(memory_dir):
    """Rebuild index reads frontmatter and generates correct entries."""
    (memory_dir / "user_role.md").write_text(
        "---\nname: User Role\ndescription: role info\ntype: user\n---\ncontent"
    )
    (memory_dir / "feedback_style.md").write_text(
        "---\nname: Code Style\ndescription: style prefs\ntype: feedback\n---\ncontent"
    )

    c = DreamConsolidator(memory_dir=memory_dir)
    c._rebuild_index()

    index = (memory_dir / MEMORY_INDEX).read_text()
    assert "User Role" in index
    assert "Code Style" in index
    assert "user_role.md" in index


def test_rebuild_index_removes_empty_index(memory_dir):
    """Rebuild removes MEMORY.md if no memory files exist."""
    index_path = memory_dir / MEMORY_INDEX
    index_path.write_text("# Old index")

    c = DreamConsolidator(memory_dir=memory_dir)
    c._rebuild_index()

    assert not index_path.exists()


# ── Frontmatter Parser ───────────────────────────────────────────


def test_parse_frontmatter_valid():
    """Parse extracts name and description from frontmatter."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("---\nname: Test\ndescription: A test memory\ntype: user\n---\ncontent")
        f.flush()
        name, desc = _parse_frontmatter(Path(f.name))
    assert name == "Test"
    assert desc == "A test memory"


def test_parse_frontmatter_no_frontmatter():
    """Parse returns empty strings when no frontmatter."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Just some content")
        f.flush()
        name, desc = _parse_frontmatter(Path(f.name))
    assert name == ""
    assert desc == ""


# ── Full Pipeline ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_dream_cycle(memory_dir, mock_llm):
    """Full dream cycle: orient → gather → consolidate → prune."""
    mock_llm.generate.return_value = LLMResponse(
        content=json.dumps({
            "updates": [
                {
                    "action": "create",
                    "filename": "user_role.md",
                    "content": "---\nname: Role\ndescription: user is a dev\ntype: user\n---\nSenior dev",
                },
                {
                    "action": "create",
                    "filename": "feedback_testing.md",
                    "content": "---\nname: Testing\ndescription: testing prefs\ntype: feedback\n---\nUse real DB",
                },
            ],
            "index_lines": [
                "- [Role](user_role.md) — user is a dev",
                "- [Testing](feedback_testing.md) — testing prefs",
            ],
        }),
        finish_reason="stop",
    )

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="I'm a senior dev"),
        Message(role="assistant", content="Got it!"),
        Message(role="user", content="Always use real databases in tests"),
        Message(role="assistant", content="Will do!"),
    ]

    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    result = await c.run(messages)

    assert result.total_changes == 2
    assert "user_role.md" in result.created
    assert "feedback_testing.md" in result.created
    assert (memory_dir / "user_role.md").exists()
    assert (memory_dir / "feedback_testing.md").exists()
    assert (memory_dir / MEMORY_INDEX).exists()

    index = (memory_dir / MEMORY_INDEX).read_text()
    assert "Role" in index
    assert "Testing" in index


@pytest.mark.asyncio
async def test_dream_nothing_to_remember(memory_dir, mock_llm):
    """Dream handles case where nothing is worth remembering."""
    mock_llm.generate.return_value = LLMResponse(
        content=json.dumps({"updates": [], "index_lines": []}),
        finish_reason="stop",
    )

    messages = [
        Message(role="user", content="What's 2+2?"),
        Message(role="assistant", content="4"),
    ]

    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    result = await c.run(messages)

    assert result.total_changes == 0


# ── Hook Integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dream_registers_with_hook_registry(memory_dir, mock_llm):
    """DreamConsolidator.register() wires up SESSION_END hook."""
    mock_llm.generate.return_value = LLMResponse(
        content=json.dumps({"updates": [], "index_lines": []}),
        finish_reason="stop",
    )

    registry = HookRegistry()
    c = DreamConsolidator(memory_dir=memory_dir, llm_client=mock_llm)
    c.register(registry)

    payload = SessionEndPayload(
        messages=[Message(role="user", content="test")],
        final_event_type="agent_done",
        steps=1,
    )
    await registry.emit(HookEvent.SESSION_END, payload)

    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_no_llm_skips_silently(memory_dir):
    """on_session_end is a no-op when no LLM is configured."""
    c = DreamConsolidator(memory_dir=memory_dir, llm_client=None)
    payload = SessionEndPayload(
        messages=[Message(role="user", content="test")],
        final_event_type="agent_done",
        steps=1,
    )
    # Should not raise
    await c.on_session_end(payload)


# ── DreamResult ──────────────────────────────────────────────────


def test_dream_result_total_changes():
    """DreamResult.total_changes sums all operations."""
    r = DreamResult(created=["a.md"], updated=["b.md"], deleted=["c.md", "d.md"])
    assert r.total_changes == 4


def test_dream_result_repr():
    """DreamResult has readable repr."""
    r = DreamResult(created=["a.md"])
    assert "a.md" in repr(r)


def test_dream_result_defaults():
    """DreamResult defaults to empty lists."""
    r = DreamResult()
    assert r.total_changes == 0
