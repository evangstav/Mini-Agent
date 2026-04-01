"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Optional

from .context import SystemPromptBuilder, ToolResultStore, compact_messages, compute_compact_threshold
from .events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .hooks import CompactPayload, HookEvent, HookRegistry, SessionEndPayload, SessionStartPayload
from .llm import LLMClient
from .schema import Message
from .tools.base import Tool, ToolResult

# Type alias for the permission callback
PermissionCallbackType = Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]]


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
        permission_callback: PermissionCallbackType | None = None,
        hooks: HookRegistry | None = None,
        session_id: str | None = None,
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
        self.hooks = hooks or HookRegistry()
        self.session_id = session_id
        self._running = False
        self._session_started = False

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    async def run_stream(
        self, cancel_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent loop, yielding events as they occur.

        Yields:
            AgentEvent subclasses: TextChunk, ThinkingChunk, ToolStart,
            ToolEnd, PermissionRequest, AgentDone, AgentCancelled, AgentError.
        """
        if self._running:
            raise RuntimeError("run_stream() is already running; concurrent calls are not supported")
        self._running = True

        try:
            # Emit session start hook only once per session (not per turn)
            if not self._session_started:
                self._session_started = True
                await self.hooks.emit(
                    HookEvent.SESSION_START,
                    SessionStartPayload(
                        session_id=self.session_id,
                        system_prompt=self.system_prompt,
                    ),
                )

            for step in range(self.max_steps):
                if cancel_event and cancel_event.is_set():
                    yield AgentCancelled(content="Task cancelled by user.", steps=step)
                    return

                # Compact context if it's getting large
                if self.compact_threshold > 0:
                    old_count = len(self.messages)
                    self.messages = await compact_messages(
                        self.messages, self.llm, self.compact_threshold
                    )
                    new_count = len(self.messages)
                    if new_count < old_count:
                        # Compaction happened — fire the COMPACT hook
                        summary_text = ""
                        if new_count > 1 and self.messages[1].role == "assistant":
                            content = self.messages[1].content
                            if isinstance(content, str):
                                summary_text = content
                        await self.hooks.emit(
                            HookEvent.COMPACT,
                            CompactPayload(
                                old_turns_count=old_count - new_count,
                                summary_text=summary_text,
                            ),
                        )

                # Think: call LLM with streaming
                try:
                    response = None
                    async for delta in self.llm.generate_stream(
                        messages=self.messages, tools=list(self.tools.values())
                    ):
                        if delta.type == "text_delta":
                            yield TextChunk(content=delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingChunk(content=delta.text)
                        elif delta.type == "message_complete":
                            response = delta.response

                    if response is None:
                        yield AgentError(
                            error="LLM stream ended without message_complete", steps=step
                        )
                        return
                except Exception as e:
                    yield AgentError(error=f"LLM call failed: {e}", steps=step)
                    return

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

                # Act: execute tool calls in parallel, Observe: collect results
                tool_tasks: list[tuple[str, str, dict]] = []  # (id, name, args)
                permission_denied: dict[str, ToolResult] = {}

                for tool_call in response.tool_calls:
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    # Permission gating (serial — requires user interaction)
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
                            denied_result = ToolResult(
                                success=False,
                                content="",
                                error=f"Permission denied for tool: {function_name}",
                            )
                            permission_denied[tool_call_id] = denied_result
                            yield ToolEnd(
                                tool_call_id=tool_call_id,
                                tool_name=function_name,
                                success=False,
                                content="",
                                error=denied_result.error,
                            )
                            self.messages.append(
                                Message(
                                    role="tool",
                                    content=f"Error: {denied_result.error}",
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
                    tool_tasks.append((tool_call_id, function_name, arguments))

                # Execute permitted tool calls in parallel
                async def _execute_tool(tc_id: str, fn_name: str, args: dict) -> tuple[str, str, ToolResult]:
                    if fn_name not in self.tools:
                        return tc_id, fn_name, ToolResult(
                            success=False, content="", error=f"Unknown tool: {fn_name}"
                        )
                    try:
                        return tc_id, fn_name, await self.tools[fn_name].execute(**args)
                    except Exception as e:
                        return tc_id, fn_name, ToolResult(
                            success=False, content="", error=f"Tool execution failed: {e}"
                        )

                if tool_tasks:
                    results = await asyncio.gather(
                        *(_execute_tool(tc_id, fn, args) for tc_id, fn, args in tool_tasks)
                    )

                    for tool_call_id, function_name, result in results:
                        # Store large results to disk if configured
                        content = result.content if result.success else f"Error: {result.error}"
                        if self.tool_result_store and result.success:
                            content = self.tool_result_store.store_if_large(content, tool_call_id)

                        yield ToolEnd(
                            tool_call_id=tool_call_id,
                            tool_name=function_name,
                            success=result.success,
                            content=content if result.success else "",
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
                    yield AgentCancelled(content="Task cancelled by user.", steps=step + 1)
                    return

            yield AgentError(
                error=f"Task couldn't be completed after {self.max_steps} steps.",
                steps=self.max_steps,
            )
        finally:
            self._running = False

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        Consumes run_stream() and returns the final text result.
        Backward-compatible with the original synchronous-return API.
        """
        async for event in self.run_stream(cancel_event=cancel_event):
            if isinstance(event, AgentDone):
                return event.content
            elif isinstance(event, AgentCancelled):
                return event.content
            elif isinstance(event, AgentError):
                return event.error
        return ""

    async def end_session(self) -> None:
        """Emit SESSION_END hook. Call once when the session truly ends (e.g. REPL exit)."""
        if not self._session_started:
            return
        await self.hooks.emit(
            HookEvent.SESSION_END,
            SessionEndPayload(
                session_id=self.session_id,
                messages=self.messages.copy(),
                final_event_type="session_end",
                steps=0,
            ),
        )
        self._session_started = False

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
