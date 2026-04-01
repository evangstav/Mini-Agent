"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Optional

from .context import SystemPromptBuilder, ToolResultStore, compact_messages, compute_compact_threshold
from .events import (
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    SubAgentDone,
    SubAgentError,
    SubAgentSpawn,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .llm import LLMClient
from .schema import Message
from .tools.base import Tool, ToolResult


class Agent:
    """Single agent with basic tools and MCP support.

    Supports two execution modes:
    - run(): Returns final text (backward-compatible)
    - run_stream(): Async generator yielding AgentEvent objects
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        tool_result_store: ToolResultStore | None = None,
        compact_threshold: int | None = None,
        context_window: int = 200_000,
        compact_threshold_pct: float = 0.85,
        compaction_reserve: int = 20_000,
        project_dir: str | None = None,
        permission_callback: object = None,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.tool_result_store = tool_result_store
        self.context_window = context_window
        self.compact_threshold_pct = compact_threshold_pct
        self.compaction_reserve = compaction_reserve
        # Explicit threshold overrides percentage calculation
        self.compact_threshold = (
            compact_threshold
            if compact_threshold is not None
            else compute_compact_threshold(context_window, compact_threshold_pct, compaction_reserve)
        )

        # Build system prompt with CLAUDE.md and git info if project_dir given
        if project_dir:
            builder = SystemPromptBuilder(system_prompt, project_dir)
            system_prompt = builder.build()

        self.system_prompt = system_prompt
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]
        self._permission_callback = permission_callback

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    async def run_stream(
        self, cancel_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent loop, yielding events as they occur.

        Yields:
            AgentEvent subclasses: TextChunk, ThinkingChunk, ToolStart,
            ToolEnd, PermissionRequest, AgentDone, AgentError.
        """
        for step in range(self.max_steps):
            if cancel_event and cancel_event.is_set():
                yield AgentDone(content="Task cancelled by user.", steps=step)
                return

            # Compact context if it's getting large
            if self.compact_threshold > 0:
                self.messages = await compact_messages(
                    self.messages, self.llm, self.compact_threshold
                )

            # Think: call LLM
            try:
                response = await self.llm.generate(
                    messages=self.messages, tools=list(self.tools.values())
                )
            except Exception as e:
                yield AgentError(error=f"LLM call failed: {e}", steps=step)
                return

            # Emit thinking if present
            if response.thinking:
                yield ThinkingChunk(content=response.thinking)

            # Emit text content if present
            if response.content:
                yield TextChunk(content=response.content)

            # Record assistant message
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            # No tool calls → task complete
            if not response.tool_calls:
                yield AgentDone(content=response.content, steps=step + 1)
                return

            # Act: execute tool calls, Observe: collect results
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

                # Permission gating
                if self._permission_callback is not None:
                    yield PermissionRequest(
                        tool_call_id=tool_call_id,
                        tool_name=function_name,
                        arguments=arguments,
                    )
                    granted = await self._permission_callback(
                        function_name, arguments
                    )
                    if not granted:
                        result = ToolResult(
                            success=False,
                            content="",
                            error=f"Permission denied for tool: {function_name}",
                        )
                        yield ToolEnd(
                            tool_call_id=tool_call_id,
                            tool_name=function_name,
                            success=False,
                            content="",
                            error=result.error,
                        )
                        self.messages.append(
                            Message(
                                role="tool",
                                content=f"Error: {result.error}",
                                tool_call_id=tool_call_id,
                                name=function_name,
                            )
                        )
                        continue

                yield ToolStart(
                    tool_call_id=tool_call_id,
                    tool_name=function_name,
                    arguments=arguments,
                )

                if function_name not in self.tools:
                    result = ToolResult(
                        success=False, content="", error=f"Unknown tool: {function_name}"
                    )
                else:
                    try:
                        result = await self.tools[function_name].execute(**arguments)
                    except Exception as e:
                        result = ToolResult(
                            success=False, content="", error=f"Tool execution failed: {e}"
                        )

                # Store large results to disk if configured
                content = result.content if result.success else f"Error: {result.error}"
                if self.tool_result_store and result.success:
                    content = self.tool_result_store.store_if_large(content, tool_call_id)

                yield ToolEnd(
                    tool_call_id=tool_call_id,
                    tool_name=function_name,
                    success=result.success,
                    content=result.content if result.success else "",
                    error=result.error,
                )

                self.messages.append(
                    Message(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call_id,
                        name=function_name,
                    )
                )

                if cancel_event and cancel_event.is_set():
                    yield AgentDone(content="Task cancelled by user.", steps=step + 1)
                    return

        yield AgentError(
            error=f"Task couldn't be completed after {self.max_steps} steps.",
            steps=self.max_steps,
        )

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        Consumes run_stream() and returns the final text result.
        Backward-compatible with the original synchronous-return API.
        """
        async for event in self.run_stream(cancel_event=cancel_event):
            if isinstance(event, AgentDone):
                return event.content
            elif isinstance(event, AgentError):
                return event.error
        return ""

    def fork(
        self,
        task: str,
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
    ) -> "Agent":
        """Fork this agent to run a sub-task with shared prompt cache prefix.

        Creates a new Agent sharing the same system prompt (byte-identical for
        cache hits on providers that support prompt caching) and LLM client,
        but with cloned message history for isolated mutable state.

        Args:
            task: User message describing the sub-task.
            tools: Optional tool list override; defaults to parent's tools.
            max_steps: Optional max steps override; defaults to parent's value.

        Returns:
            A new Agent ready to execute via run() or run_stream().
        """
        forked = Agent.__new__(Agent)
        # Share immutable/stateless references for cache alignment
        forked.llm = self.llm
        forked.system_prompt = self.system_prompt
        forked.tools = dict(self.tools) if tools is None else {t.name: t for t in tools}
        forked.max_steps = max_steps if max_steps is not None else self.max_steps
        forked.tool_result_store = self.tool_result_store
        forked.compact_threshold = self.compact_threshold
        forked.context_window = self.context_window
        forked.compact_threshold_pct = self.compact_threshold_pct
        forked.compaction_reserve = self.compaction_reserve
        forked._permission_callback = self._permission_callback

        # Clone message history — deep copy so mutations are isolated
        forked.messages = [msg.model_copy(deep=True) for msg in self.messages]

        # Add the sub-task
        forked.add_user_message(task)
        return forked

    async def run_forked(
        self,
        tasks: list[str],
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
    ) -> list[tuple[str, str]]:
        """Run multiple sub-tasks in parallel via forked agents.

        Each task gets its own forked agent sharing this agent's prompt cache
        prefix. All forks run concurrently via asyncio.gather.

        Args:
            tasks: List of task descriptions (one sub-agent per task).
            tools: Optional tool list override for all forks.
            max_steps: Optional max steps override for all forks.

        Returns:
            List of (agent_id, result) tuples in the same order as tasks.
        """
        agents: list[tuple[str, "Agent"]] = []
        for task in tasks:
            agent_id = uuid.uuid4().hex[:8]
            forked = self.fork(task, tools=tools, max_steps=max_steps)
            agents.append((agent_id, forked))

        async def _run_one(agent_id: str, agent: "Agent") -> tuple[str, str]:
            result = await agent.run()
            return (agent_id, result)

        results = await asyncio.gather(
            *[_run_one(aid, a) for aid, a in agents],
            return_exceptions=True,
        )

        out: list[tuple[str, str]] = []
        for i, r in enumerate(results):
            agent_id = agents[i][0]
            if isinstance(r, BaseException):
                out.append((agent_id, f"Sub-agent error: {r}"))
            else:
                out.append(r)
        return out

    async def run_forked_stream(
        self,
        tasks: list[str],
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run multiple sub-tasks in parallel, yielding lifecycle events.

        Yields SubAgentSpawn when each fork starts, SubAgentDone/SubAgentError
        when each completes. Useful for observability in streaming UIs.

        Args:
            tasks: List of task descriptions.
            tools: Optional tool list override.
            max_steps: Optional max steps override.
        """
        agents: list[tuple[str, "Agent"]] = []
        for task in tasks:
            agent_id = uuid.uuid4().hex[:8]
            forked = self.fork(task, tools=tools, max_steps=max_steps)
            agents.append((agent_id, forked))

        # Emit spawn events
        for i, (agent_id, _) in enumerate(agents):
            yield SubAgentSpawn(agent_id=agent_id, task=tasks[i])

        async def _run_one(agent_id: str, agent: "Agent") -> tuple[str, str]:
            return (agent_id, await agent.run())

        results = await asyncio.gather(
            *[_run_one(aid, a) for aid, a in agents],
            return_exceptions=True,
        )

        for i, r in enumerate(results):
            agent_id = agents[i][0]
            if isinstance(r, BaseException):
                yield SubAgentError(agent_id=agent_id, error=str(r))
            else:
                yield SubAgentDone(agent_id=r[0], result=r[1])

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
