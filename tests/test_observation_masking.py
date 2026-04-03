"""Tests for observation masking (NeurIPS 2025 research-backed feature)."""

import pytest

from mini_agent.context import mask_observations
from mini_agent.schema import Message


class TestMaskObservations:
    def _make_conversation(self, tool_results: int = 5, recent: int = 2) -> list[Message]:
        """Build a conversation with system + alternating assistant/tool messages."""
        msgs = [Message(role="system", content="System prompt")]
        for i in range(tool_results):
            msgs.append(Message(role="assistant", content=f"Using tool {i}", tool_calls=[]))
            msgs.append(Message(
                role="tool",
                content=f"Tool output {i} with lots of detail " * 100,  # ~3000 chars, above snip threshold
                tool_call_id=f"call_{i}",
                name=f"tool_{i}",
            ))
        # Add recent user + assistant
        msgs.append(Message(role="user", content="Final question"))
        msgs.append(Message(role="assistant", content="Final answer"))
        return msgs

    def test_masks_old_tool_outputs(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=4)

        # Old tool messages should be masked
        tool_msgs = [m for m in result if m.role == "tool"]
        masked = [m for m in tool_msgs if "snipped" in m.content.lower()]
        assert len(masked) > 0, "Expected some tool outputs to be masked"

    def test_preserves_recent_tool_outputs(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=4)

        # The last few tool messages should NOT be masked
        recent_tools = [m for m in result[-4:] if m.role == "tool"]
        for m in recent_tools:
            assert "masked" not in m.content.lower(), f"Recent tool output should not be masked: {m.content[:50]}"

    def test_preserves_assistant_messages(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=4)

        # ALL assistant messages should be preserved in full
        original_assistant = [m for m in msgs if m.role == "assistant"]
        result_assistant = [m for m in result if m.role == "assistant"]
        assert len(original_assistant) == len(result_assistant)
        for orig, res in zip(original_assistant, result_assistant):
            assert orig.content == res.content

    def test_preserves_user_messages(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=4)

        original_user = [m for m in msgs if m.role == "user"]
        result_user = [m for m in result if m.role == "user"]
        assert len(original_user) == len(result_user)

    def test_preserves_system_message(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=4)

        assert result[0].role == "system"
        assert result[0].content == "System prompt"

    def test_preserves_tool_call_metadata(self):
        msgs = self._make_conversation(tool_results=5)
        result = mask_observations(msgs, keep_recent=6)

        masked_tools = [m for m in result if m.role == "tool" and "snipped" in m.content.lower()]
        for m in masked_tools:
            assert m.tool_call_id is not None, "Masked tool should keep tool_call_id"
            assert m.name is not None, "Masked tool should keep name"

    def test_short_conversation_unchanged(self):
        msgs = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = mask_observations(msgs, keep_recent=6)
        assert len(result) == len(msgs)
        assert all(r.content == o.content for r, o in zip(result, msgs))

    def test_empty_messages(self):
        result = mask_observations([], keep_recent=6)
        assert result == []

    def test_snipped_content_keeps_beginning_and_end(self):
        """Snipped tool output should keep first half + last quarter."""
        long_content = "START " + "x" * 3000 + " END"
        msgs = [
            Message(role="system", content="System"),
            Message(role="assistant", content="Using tool"),
            Message(role="tool", content=long_content, tool_call_id="c1", name="t1"),
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
        ]
        result = mask_observations(msgs, keep_recent=2)
        snipped = [m for m in result if m.role == "tool" and "snipped" in m.content.lower()]
        assert len(snipped) == 1
        assert snipped[0].content.startswith("START"), "Should keep the beginning"
        assert snipped[0].content.endswith(" END"), "Should keep the end"
        assert len(snipped[0].content) < len(long_content), "Should be shorter than original"
