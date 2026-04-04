"""MessageLog — append-only ordered message history."""

import json
import logging
from typing import Any

from .schema import Message, ToolCall

logger = logging.getLogger(__name__)


class MessageLog:
    """Append-only ordered message history with role-based operations.

    Follows the MessageLog concept: init, append, replace_prefix, clear, fork.
    """

    def __init__(self, system_prompt: str) -> None:
        self._messages: list[Message] = [Message(role="system", content=system_prompt)]

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def system_message(self) -> Message:
        return self._messages[0]

    def __len__(self) -> int:
        return len(self._messages)

    def append(self, msg: Message) -> None:
        self._messages.append(msg)

    def append_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def append_assistant(
        self,
        content: str,
        thinking: str | None = None,
        thinking_signature: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self._messages.append(Message(
            role="assistant",
            content=content,
            thinking=thinking,
            thinking_signature=thinking_signature,
            tool_calls=tool_calls,
        ))

    def append_tool_result(self, content: str, tool_call_id: str, name: str) -> None:
        self._messages.append(Message(
            role="tool", content=content, tool_call_id=tool_call_id, name=name,
        ))

    def extend(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    def replace_prefix(self, new_messages: list[Message]) -> None:
        """Replace entire message list (used after compaction)."""
        self._messages = new_messages

    def clear(self) -> None:
        """Clear all messages except the system prompt."""
        self._messages = [self._messages[0]]

    def copy(self) -> list[Message]:
        return self._messages.copy()

    def fork(self, rewind: int = 0) -> "MessageLog":
        """Create an independent copy, optionally rewinding N turn pairs."""
        messages = self._messages.copy()
        if rewind > 0:
            pairs_removed = 0
            while pairs_removed < rewind and len(messages) > 1:
                if messages[-1].role in ("tool", "assistant"):
                    messages.pop()
                elif messages[-1].role == "user":
                    messages.pop()
                    pairs_removed += 1
                else:
                    break

        new_log = MessageLog.__new__(MessageLog)
        new_log._messages = messages
        return new_log

    def snip_observations(self, keep_recent: int = 6, snip_threshold: int = 2000) -> None:
        """Snip old tool outputs in-place, keeping beginning + end.

        Matches the MessageLog concept's snip_observation action.
        Only modifies tool messages older than keep_recent that exceed snip_threshold.
        """
        if len(self._messages) <= keep_recent + 1:
            return

        system_offset = 1 if self._messages and self._messages[0].role == "system" else 0
        conversation = self._messages[system_offset:]

        if len(conversation) <= keep_recent:
            return

        boundary = len(conversation) - keep_recent

        for i in range(boundary):
            msg = conversation[i]
            if msg.role != "tool":
                continue
            content = msg.content if isinstance(msg.content, str) else ""
            if len(content) <= snip_threshold:
                continue

            # Snip: keep first half + last quarter
            first_half = len(content) // 2
            last_quarter = len(content) // 4
            snipped = len(content) - first_half - last_quarter
            new_content = (
                content[:first_half]
                + f"\n[... {snipped} chars snipped ...]\n"
                + content[-last_quarter:]
            )
            # Replace in-place
            self._messages[system_offset + i] = Message(
                role=msg.role,
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

    def estimate_tokens(self) -> int:
        """Conservative token estimate: ~4 chars per token (tuned for code-heavy context)."""
        total_chars = 0
        for msg in self._messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                total_chars += len(json.dumps(msg.content))
            if msg.thinking:
                total_chars += len(msg.thinking)
            if msg.tool_calls:
                total_chars += len(json.dumps([tc.model_dump() for tc in msg.tool_calls]))
        return total_chars // 4
