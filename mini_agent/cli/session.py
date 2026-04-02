"""Session persistence — save/load conversation to JSON."""

import json
from pathlib import Path

from ..schema import Message


def save_session(messages: list[Message], path: str) -> None:
    """Save conversation to JSON."""
    data = [msg.model_dump() for msg in messages]
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_session(path: str) -> list[Message]:
    """Load conversation from JSON."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupt session file (invalid JSON): {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"Corrupt session file: expected a JSON array, got {type(raw).__name__}")

    messages: list[Message] = []
    for i, item in enumerate(raw):
        try:
            messages.append(Message(**item))
        except Exception as e:
            raise ValueError(f"Corrupt session file: invalid message at index {i}: {e}") from e
    return messages
