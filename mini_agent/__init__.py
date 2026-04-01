"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .context import SystemPromptBuilder, ToolResultStore, compact_messages, estimate_tokens, prune_tool_results
from .dream import DreamConsolidator, DreamResult
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
from .hooks import HookEvent, HookRegistry, SessionEndPayload, SessionStartPayload
from .llm import LLMClient
from .schema import FunctionCall, LLMProvider, LLMResponse, Message, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentDone",
    "AgentError",
    "AgentEvent",
    "DreamConsolidator",
    "DreamResult",
    "HookEvent",
    "HookRegistry",
    "LLMClient",
    "LLMProvider",
    "Message",
    "LLMResponse",
    "PermissionRequest",
    "SessionEndPayload",
    "SessionStartPayload",
    "TextChunk",
    "ThinkingChunk",
    "ToolCall",
    "ToolEnd",
    "ToolStart",
    "FunctionCall",
    "ToolResultStore",
    "SystemPromptBuilder",
    "compact_messages",
    "estimate_tokens",
    "prune_tool_results",
]
