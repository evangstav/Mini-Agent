"""Agent event types for generator-based streaming."""

from typing import Any

from pydantic import BaseModel


class AgentEvent(BaseModel):
    """Base class for all agent events."""

    type: str


class TextChunk(AgentEvent):
    """Streaming text content from LLM response."""

    type: str = "text_chunk"
    content: str


class ThinkingChunk(AgentEvent):
    """Extended thinking content from LLM response."""

    type: str = "thinking_chunk"
    content: str


class ToolStart(AgentEvent):
    """Tool execution is starting."""

    type: str = "tool_start"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolEnd(AgentEvent):
    """Tool execution completed."""

    type: str = "tool_end"
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

    type: str = "permission_request"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class AgentDone(AgentEvent):
    """Agent completed its work."""

    type: str = "agent_done"
    content: str
    steps: int


class AgentError(AgentEvent):
    """Agent encountered an error."""

    type: str = "agent_error"
    error: str
    steps: int
