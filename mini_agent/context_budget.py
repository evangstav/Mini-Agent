"""ContextBudget — monitors token usage and triggers compaction."""

import logging
from typing import Any

from .hooks import CompactPayload, HookEvent, HookRegistry
from .message_log import MessageLog
from .schema import Message

logger = logging.getLogger(__name__)


def compute_compact_threshold(
    context_window: int = 200_000,
    compact_threshold_pct: float = 0.85,
    compaction_reserve: int = 20_000,
) -> int:
    """Compute the absolute token threshold for compaction."""
    return int(context_window * compact_threshold_pct) - compaction_reserve


class ContextBudget:
    """Monitors conversation size and triggers compaction when the threshold is crossed.

    Follows the ContextBudget concept: track usage, detect when compaction needed,
    compact via LLM summarization.
    """

    def __init__(
        self,
        context_window: int = 200_000,
        compact_threshold_pct: float = 0.85,
        compaction_reserve: int = 20_000,
        compact_threshold: int | None = None,
    ) -> None:
        self.threshold = (
            compact_threshold
            if compact_threshold is not None
            else compute_compact_threshold(context_window, compact_threshold_pct, compaction_reserve)
        )

    async def maybe_compact(
        self,
        log: MessageLog,
        llm_client: Any,
        hooks: HookRegistry,
    ) -> bool:
        """If over threshold, compact the message log. Returns True if compaction occurred.

        Uses a two-pass strategy based on NeurIPS 2025 research:
        1. Observation masking (free — no LLM call): replace old tool outputs with placeholders
        2. LLM summarization (expensive): only if masking wasn't enough
        """
        if self.threshold <= 0:
            return False

        from .context import compact_messages

        # Pass 1: Observation snipping (free, no LLM call)
        log.snip_observations()

        # Pass 2: LLM compaction if still over threshold
        old_count = len(log)
        new_messages = await compact_messages(
            log.messages, llm_client, self.threshold
        )
        new_count = len(new_messages)

        if new_count < old_count:
            log.replace_prefix(new_messages)
            logger.info("Context compacted: %d → %d messages", old_count, new_count)

            summary_text = ""
            if new_count > 1 and new_messages[1].role == "assistant":
                content = new_messages[1].content
                if isinstance(content, str):
                    summary_text = content

            await hooks.emit(
                HookEvent.COMPACT,
                CompactPayload(
                    old_turns_count=old_count - new_count,
                    summary_text=summary_text,
                ),
            )
            return True
        return False
