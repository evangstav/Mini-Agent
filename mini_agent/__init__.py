"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .events import (
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .llm import LLMClient
from .schema import FunctionCall, LLMProvider, LLMResponse, Message, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentDone",
    "AgentError",
    "AgentEvent",
    "LLMClient",
    "LLMProvider",
    "Message",
    "LLMResponse",
    "PermissionRequest",
    "TextChunk",
    "ThinkingChunk",
    "ToolCall",
    "ToolEnd",
    "ToolStart",
    "FunctionCall",
]
