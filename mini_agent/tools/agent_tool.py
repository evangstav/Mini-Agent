"""Agent tool — spawn a sub-agent with focused task and tool subset.

The sub-agent shares the parent's LLM client (and therefore prompt cache
prefix) but gets its own message history.  It runs to completion and
returns a structured result to the parent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..events import AgentDone, AgentError
from ..sandbox import Sandbox
from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..agent import Agent
    from ..llm import LLMClient

logger = logging.getLogger(__name__)

# Sensible ceiling so a sub-agent can't run forever.
_DEFAULT_MAX_STEPS = 30


class AgentTool(Tool):
    """Spawn a sub-agent to handle a focused task autonomously.

    The sub-agent receives:
    - A task prompt (becomes the user message)
    - A subset of the parent's available tools (by name)
    - The parent's LLM client (shared — prompt cache friendly)
    - Its own independent message history
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        available_tools: dict[str, Tool],
        sandbox: Sandbox | None = None,
        permission_callback: Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]] | None = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        system_prompt: str | None = None,
        workspace: str | None = None,
    ):
        self._llm_client = llm_client
        self._available_tools = available_tools
        self._sandbox = sandbox
        self._permission_callback = permission_callback
        self._max_steps = max_steps
        self._workspace = workspace
        self._system_prompt = system_prompt or (
            "You are a focused sub-agent. Complete the given task using "
            "the tools available to you, then respond with your result.\n\n"
            "Rules:\n"
            "- Only state facts verified by reading files or running commands.\n"
            "- If you cannot find something, say so. Never fabricate.\n"
            "- After code changes, run tests to verify.\n"
            "- Keep your output concise and factual."
        )

    @property
    def name(self) -> str:
        return "agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a sub-agent for PARALLELIZABLE work only (searching multiple locations, "
            "running tests while editing). The sub-agent gets its own isolated context. "
            "Do NOT use for sequential coding tasks — a single agent with good context is better. "
            "Provide a detailed prompt — the sub-agent has NO prior conversation context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        tool_names = sorted(self._available_tools.keys())
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task for the sub-agent to perform. Be specific — "
                        "the sub-agent has no prior conversation context."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string", "enum": tool_names},
                    "description": (
                        f"Tools the sub-agent can use. "
                        f"Available: {', '.join(tool_names)}. "
                        f"Omit to grant all tools except 'agent' (no recursion)."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        f"Maximum agent steps (default: {self._max_steps})."
                    ),
                },
            },
            "required": ["prompt"],
        }

    async def execute(
        self,
        prompt: str,
        tools: list[str] | None = None,
        max_steps: int | None = None,
    ) -> ToolResult:
        """Spawn a sub-agent, run it to completion, return its output."""
        from ..agent import Agent  # lazy to avoid circular import

        # Resolve the tool subset
        if tools:
            missing = [t for t in tools if t not in self._available_tools]
            if missing:
                return ToolResult(
                    success=False,
                    error=f"Unknown tools: {', '.join(missing)}. "
                    f"Available: {', '.join(sorted(self._available_tools))}",
                )
            sub_tools = [self._available_tools[t] for t in tools]
        else:
            # Default: all tools except 'agent' to prevent recursive spawning
            sub_tools = [
                t for name, t in self._available_tools.items()
                if name != "agent"
            ]

        resolved_steps = max_steps or self._max_steps
        logger.info(
            "Spawning sub-agent (tools=%s, max_steps=%d)",
            [t.name for t in sub_tools],
            resolved_steps,
        )

        sub_agent = Agent(
            llm_client=self._llm_client,
            system_prompt=self._system_prompt,
            tools=sub_tools,
            max_steps=resolved_steps,
            permission_callback=self._permission_callback,
            sandbox=self._sandbox,
            project_dir=self._workspace,
        )

        sub_agent.add_user_message(prompt)

        try:
            text_chunks: list[str] = []
            async for event in sub_agent.run_stream():
                if isinstance(event, AgentDone):
                    if event.content:
                        text_chunks.append(event.content)
                elif isinstance(event, AgentError):
                    return ToolResult(
                        success=False,
                        error=f"Sub-agent error: {event.error}",
                    )
                elif hasattr(event, "content") and getattr(event, "type", "") == "text_chunk":
                    text_chunks.append(event.content)
            result_text = "\n".join(text_chunks).strip()
            return ToolResult(success=True, content=result_text or "(no output)")
        except Exception as e:
            logger.error("Sub-agent failed: %s", e)
            return ToolResult(success=False, error=f"Sub-agent exception: {e}")
