"""Agent event types for generator-based streaming."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentEvent(BaseModel):
    """Base class for all agent events."""

    type: str
    timestamp: datetime = Field(default_factory=_utc_now)


class TextChunk(AgentEvent):
    """Streaming text content from LLM response."""

    type: Literal["text_chunk"] = "text_chunk"
    content: str


class ThinkingChunk(AgentEvent):
    """Extended thinking content from LLM response."""

    type: Literal["thinking_chunk"] = "thinking_chunk"
    content: str


class ToolStart(AgentEvent):
    """Tool execution is starting."""

    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolEnd(AgentEvent):
    """Tool execution completed."""

    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    error: str | None = None


class PermissionRequest(AgentEvent):
    """Permission requested before tool execution.

    The caller can grant or deny permission by sending a response
    back through the permission callback.
    """

    type: Literal["permission_request"] = "permission_request"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class AgentDone(AgentEvent):
    """Agent completed its work."""

    type: Literal["agent_done"] = "agent_done"
    content: str
    steps: int


class AgentCancelled(AgentEvent):
    """Agent was cancelled by the user (distinct from normal completion)."""

    type: Literal["agent_cancelled"] = "agent_cancelled"
    content: str
    steps: int


class PlanProposal(AgentEvent):
    """Proposed tool calls awaiting user approval (plan mode)."""

    type: Literal["plan_proposal"] = "plan_proposal"
    proposed_calls: list[dict[str, Any]]  # [{id, name, arguments}]
    steps: int


class AgentError(AgentEvent):
    """Agent encountered an error."""

    type: Literal["agent_error"] = "agent_error"
    error: str
    steps: int
