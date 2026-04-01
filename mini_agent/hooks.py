"""Session lifecycle hooks — register callbacks for agent events."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Coroutine

from pydantic import BaseModel

from .schema import Message


class HookEvent(str, Enum):
    """Lifecycle events that hooks can subscribe to."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    COMPACT = "compact"


class SessionEndPayload(BaseModel):
    """Payload delivered to SESSION_END hooks."""

    session_id: str | None = None
    messages: list[Message]
    final_event_type: str  # "agent_done", "agent_cancelled", or "agent_error"
    steps: int
    metadata: dict[str, Any] = {}


class SessionStartPayload(BaseModel):
    """Payload delivered to SESSION_START hooks."""

    session_id: str | None = None
    system_prompt: str
    metadata: dict[str, Any] = {}


class CompactPayload(BaseModel):
    """Payload delivered to COMPACT hooks."""

    old_turns_count: int
    summary_text: str
    metadata: dict[str, Any] = {}


# Type alias for hook callbacks
HookCallback = Callable[..., Coroutine[Any, Any, None]]


class HookRegistry:
    """Registry for session lifecycle hooks.

    Usage:
        registry = HookRegistry()
        registry.on(HookEvent.SESSION_END, my_handler)
        await registry.emit(HookEvent.SESSION_END, payload)
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = defaultdict(list)

    def on(self, event: HookEvent, callback: HookCallback) -> None:
        """Register a callback for an event."""
        self._hooks[event].append(callback)

    def off(self, event: HookEvent, callback: HookCallback) -> None:
        """Remove a callback for an event."""
        try:
            self._hooks[event].remove(callback)
        except ValueError:
            pass

    async def emit(self, event: HookEvent, payload: Any) -> None:
        """Fire all registered callbacks for an event.

        Callbacks run concurrently via asyncio.gather. Errors in individual
        callbacks are caught and logged (they never crash the agent).
        The callback list is snapshotted before gather to prevent
        mutation-during-iteration bugs.
        """
        callbacks = list(self._hooks.get(event, []))
        if not callbacks:
            return

        results = await asyncio.gather(
            *(cb(payload) for cb in callbacks),
            return_exceptions=True,
        )
        # Surface exceptions as warnings (don't crash the agent)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # In production, this would use proper logging
                print(f"[hooks] {event.value} callback {i} failed: {result}")

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all hooks, or all hooks for a specific event."""
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)
