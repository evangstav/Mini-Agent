"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Optional

from .context import SystemPromptBuilder, ToolResultStore, compact_messages, compute_compact_threshold, estimate_tokens
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
from .sandbox import Decision, PermissionMode, Sandbox
from .schema import Message, TokenUsage
from .session_memory import SessionMemory
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
        sandbox: Sandbox | None = None,
        hooks: HookRegistry | None = None,
        session_id: str | None = None,
        session_memory: SessionMemory | None = None,
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
        self.sandbox = sandbox or Sandbox(PermissionMode.AUTO)
        self.hooks = hooks or HookRegistry()
        self.session_id = session_id
        self._running = False
        self.token_usage = TokenUsage()  # Accumulated token usage from API responses
        self.session_memory = session_memory

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
            # Emit session start hook
            await self.hooks.emit(
                HookEvent.SESSION_START,
                SessionStartPayload(
                    session_id=self.session_id,
                    system_prompt=self.system_prompt,
                ),
            )

            for step in range(self.max_steps):
                if cancel_event and cancel_event.is_set():
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step)
                    await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

                # Compact context if it's getting large
                if self.compact_threshold > 0:
                    current_tokens = estimate_tokens(self.messages)
                    old_count = len(self.messages)
                    used_free_summary = False

                    # Try session memory as free compaction summary first
                    if (
                        self.session_memory
                        and current_tokens >= self.compact_threshold
                    ):
                        free_summary = self.session_memory.as_compaction_summary()
                        if free_summary:
                            system_msg = self.messages[0] if self.messages and self.messages[0].role == "system" else None
                            keep_recent = 6
                            conversation = self.messages[1:] if system_msg else self.messages
                            if len(conversation) > keep_recent:
                                recent = conversation[-keep_recent:]
                                compacted: list[Message] = []
                                if system_msg:
                                    compacted.append(system_msg)
                                compacted.append(Message(
                                    role="user",
                                    content=f"[Session memory summary of {old_count - keep_recent - (1 if system_msg else 0)} earlier messages]:\n{free_summary}",
                                ))
                                compacted.extend(recent)
                                self.messages = compacted
                                used_free_summary = True

                    # Fall back to LLM-based compaction
                    if not used_free_summary:
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
                        error_event = AgentError(
                            error="LLM stream ended without message_complete", steps=step
                        )
                        await self._emit_session_end(error_event)
                        yield error_event
                        return
                except Exception as e:
                    error_event = AgentError(error=f"LLM call failed: {e}", steps=step)
                    await self._emit_session_end(error_event)
                    yield error_event
                    return

                # Accumulate token usage from API response
                if response.usage:
                    self.token_usage.prompt_tokens += response.usage.prompt_tokens
                    self.token_usage.completion_tokens += response.usage.completion_tokens
                    self.token_usage.total_tokens += response.usage.total_tokens

                # Record assistant message
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=response.tool_calls,
                )
                self.messages.append(assistant_msg)

                # Track assistant text in session memory
                if self.session_memory and response.content and not response.tool_calls:
                    self.session_memory.record_assistant_text(response.content)

                # No tool calls → task complete
                if not response.tool_calls:
                    done_event = AgentDone(content=response.content, steps=step + 1)
                    await self._emit_session_end(done_event)
                    yield done_event
                    return

                # Act: execute tool calls in parallel, Observe: collect results
                # Separate permission-gated calls (must be serial) from executable ones
                pending_events: list[AgentEvent] = []
                tool_tasks: list[tuple[str, str, dict]] = []  # (id, name, args)
                permission_denied: dict[str, ToolResult] = {}

                for tool_call in response.tool_calls:
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    # Sandbox decision
                    decision = self.sandbox.check(function_name, arguments)

                    if decision == Decision.DENY:
                        denied_result = ToolResult(
                            success=False,
                            content="",
                            error=f"Tool blocked by sandbox ({self.sandbox.mode.value} mode): {function_name}",
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

                    if decision == Decision.ASK and self._permission_callback is not None:
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

                        # Record in session memory
                        if self.session_memory:
                            # Find the original arguments for this tool call
                            tc_args = {}
                            for tc_id, fn, args in tool_tasks:
                                if tc_id == tool_call_id:
                                    tc_args = args
                                    break
                            self.session_memory.record_tool_call(
                                function_name, tc_args, content, result.success,
                            )

                    # Write session memory snapshot if due
                    if self.session_memory:
                        token_est = estimate_tokens(self.messages)
                        if self.session_memory.should_update(token_est):
                            self.session_memory.write_snapshot(token_est)

                if cancel_event and cancel_event.is_set():
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step + 1)
                    await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

            error_event = AgentError(
                error=f"Task couldn't be completed after {self.max_steps} steps.",
                steps=self.max_steps,
            )
            await self._emit_session_end(error_event)
            yield error_event
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

    async def _emit_session_end(self, event: AgentDone | AgentCancelled | AgentError) -> None:
        """Emit SESSION_END hook with the conversation history."""
        if isinstance(event, AgentCancelled):
            event_type = "agent_cancelled"
        elif isinstance(event, AgentDone):
            event_type = "agent_done"
        else:
            event_type = "agent_error"

        await self.hooks.emit(
            HookEvent.SESSION_END,
            SessionEndPayload(
                session_id=self.session_id,
                messages=self.messages.copy(),
                final_event_type=event_type,
                steps=event.steps,
            ),
        )

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
