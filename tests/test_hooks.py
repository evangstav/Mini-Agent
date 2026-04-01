"""Test cases for the hooks module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mini_agent.hooks import (
    CompactPayload,
    HookEvent,
    HookRegistry,
    SessionEndPayload,
    SessionStartPayload,
)
from mini_agent.schema import Message


@pytest.mark.asyncio
async def test_registry_emits_to_registered_callback():
    """Registered callback receives the payload."""
    registry = HookRegistry()
    callback = AsyncMock()
    registry.on(HookEvent.SESSION_END, callback)

    payload = SessionEndPayload(
        messages=[Message(role="user", content="hi")],
        final_event_type="agent_done",
        steps=1,
    )
    await registry.emit(HookEvent.SESSION_END, payload)

    callback.assert_awaited_once_with(payload)


@pytest.mark.asyncio
async def test_registry_emits_to_multiple_callbacks():
    """Multiple callbacks for the same event all fire."""
    registry = HookRegistry()
    cb1 = AsyncMock()
    cb2 = AsyncMock()
    registry.on(HookEvent.SESSION_END, cb1)
    registry.on(HookEvent.SESSION_END, cb2)

    payload = SessionEndPayload(
        messages=[], final_event_type="agent_done", steps=0
    )
    await registry.emit(HookEvent.SESSION_END, payload)

    cb1.assert_awaited_once()
    cb2.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_no_callbacks_is_noop():
    """Emitting with no callbacks doesn't raise."""
    registry = HookRegistry()
    payload = SessionStartPayload(system_prompt="test")
    await registry.emit(HookEvent.SESSION_START, payload)


@pytest.mark.asyncio
async def test_registry_off_removes_callback():
    """Removing a callback prevents it from firing."""
    registry = HookRegistry()
    callback = AsyncMock()
    registry.on(HookEvent.SESSION_END, callback)
    registry.off(HookEvent.SESSION_END, callback)

    payload = SessionEndPayload(
        messages=[], final_event_type="agent_done", steps=0
    )
    await registry.emit(HookEvent.SESSION_END, payload)

    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_off_nonexistent_is_noop():
    """Removing a callback that was never registered doesn't raise."""
    registry = HookRegistry()
    callback = AsyncMock()
    registry.off(HookEvent.SESSION_START, callback)


@pytest.mark.asyncio
async def test_registry_callback_error_doesnt_crash(caplog):
    """A failing callback doesn't prevent other callbacks from running."""
    registry = HookRegistry()

    async def failing_cb(payload):
        raise ValueError("boom")

    good_cb = AsyncMock()

    registry.on(HookEvent.SESSION_END, failing_cb)
    registry.on(HookEvent.SESSION_END, good_cb)

    payload = SessionEndPayload(
        messages=[], final_event_type="agent_done", steps=0
    )
    await registry.emit(HookEvent.SESSION_END, payload)

    good_cb.assert_awaited_once()
    assert "boom" in caplog.text


@pytest.mark.asyncio
async def test_registry_clear_all():
    """clear() with no args removes all hooks."""
    registry = HookRegistry()
    cb = AsyncMock()
    registry.on(HookEvent.SESSION_START, cb)
    registry.on(HookEvent.SESSION_END, cb)

    registry.clear()

    await registry.emit(HookEvent.SESSION_START, SessionStartPayload(system_prompt=""))
    await registry.emit(
        HookEvent.SESSION_END,
        SessionEndPayload(messages=[], final_event_type="agent_done", steps=0),
    )
    cb.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_clear_specific_event():
    """clear(event) only removes hooks for that event."""
    registry = HookRegistry()
    start_cb = AsyncMock()
    end_cb = AsyncMock()
    registry.on(HookEvent.SESSION_START, start_cb)
    registry.on(HookEvent.SESSION_END, end_cb)

    registry.clear(HookEvent.SESSION_START)

    await registry.emit(HookEvent.SESSION_START, SessionStartPayload(system_prompt=""))
    await registry.emit(
        HookEvent.SESSION_END,
        SessionEndPayload(messages=[], final_event_type="agent_done", steps=0),
    )

    start_cb.assert_not_awaited()
    end_cb.assert_awaited_once()


@pytest.mark.asyncio
async def test_different_events_dont_cross():
    """Callbacks only fire for their registered event."""
    registry = HookRegistry()
    start_cb = AsyncMock()
    end_cb = AsyncMock()
    registry.on(HookEvent.SESSION_START, start_cb)
    registry.on(HookEvent.SESSION_END, end_cb)

    await registry.emit(HookEvent.SESSION_START, SessionStartPayload(system_prompt=""))

    start_cb.assert_awaited_once()
    end_cb.assert_not_awaited()


def test_session_end_payload_defaults():
    """SessionEndPayload has sensible defaults."""
    payload = SessionEndPayload(
        messages=[], final_event_type="agent_done", steps=5
    )
    assert payload.session_id is None
    assert payload.metadata == {}
    assert payload.steps == 5


def test_compact_payload():
    """CompactPayload stores compaction metadata."""
    payload = CompactPayload(old_turns_count=10, summary_text="summary")
    assert payload.old_turns_count == 10
