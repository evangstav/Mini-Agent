"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .context import SystemPromptBuilder, ToolResultStore, compact_messages, estimate_tokens, extract_handoff_context, prune_tool_results
from .events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    PlanProposal,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .llm import LLMClient
from .log import setup_logging
from .permissions import PermissionRule, PermissionRuleset, RuleAction, load_rules, load_rules_from_toml
from .sandbox import Decision, PermissionMode, Sandbox
from .schema import FunctionCall, LLMProvider, LLMResponse, Message, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentCancelled",
    "AgentDone",
    "AgentError",
    "AgentEvent",
    "LLMClient",
    "LLMProvider",
    "Message",
    "PermissionMode",
    "PermissionRule",
    "PermissionRuleset",
    "RuleAction",
    "Sandbox",
    "Decision",
    "load_rules",
    "load_rules_from_toml",
    "LLMResponse",
    "PermissionRequest",
    "PlanProposal",
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
    "extract_handoff_context",
    "prune_tool_results",
    "setup_logging",
]
