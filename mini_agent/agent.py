"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .audit import AuditLogger
from .context import SystemPromptBuilder, ToolResultStore, compact_messages, compute_compact_threshold
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
from .hooks import CompactPayload, HookEvent, HookRegistry, SessionEndPayload, SessionStartPayload
from .llm import LLMClient
from .sandbox import Decision, PermissionMode, Sandbox
from .schema import Message, TokenUsage
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
        self._audit = AuditLogger(session_id) if session_id else None

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    async def run_stream(
        self,
        cancel_event: Optional[asyncio.Event] = None,
        plan_mode: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent loop, yielding events as they occur.

        Args:
            cancel_event: Set to cancel execution.
            plan_mode: If True, yield PlanProposal instead of executing tools.
                The assistant message (with tool_calls) is added to history
                so execute_plan()/reject_plan() can pick up where we left off.

        Yields:
            AgentEvent subclasses: TextChunk, ThinkingChunk, ToolStart,
            ToolEnd, PermissionRequest, PlanProposal, AgentDone,
            AgentCancelled, AgentError.
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

            logger.info("Agent run started (max_steps=%d, session=%s)", self.max_steps, self.session_id)

            for step in range(self.max_steps):
                if cancel_event and cancel_event.is_set():
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step)
                    await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

                # Compact context if it's getting large
                if self.compact_threshold > 0:
                    old_count = len(self.messages)
                    self.messages = await compact_messages(
                        self.messages, self.llm, self.compact_threshold
                    )
                    new_count = len(self.messages)
                    if new_count < old_count:
                        logger.info("Context compacted: %d → %d messages", old_count, new_count)
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

                logger.debug("Step %d: calling LLM", step)
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
                    logger.error("LLM call failed at step %d: %s", step, e)
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

                # No tool calls → task complete
                if not response.tool_calls:
                    logger.info("Agent completed at step %d", step + 1)
                    done_event = AgentDone(content=response.content, steps=step + 1)
                    await self._emit_session_end(done_event)
                    yield done_event
                    return

                # Plan mode: yield proposal instead of executing tools
                if plan_mode and response.tool_calls:
                    proposed = [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in response.tool_calls
                    ]
                    yield PlanProposal(proposed_calls=proposed, steps=step + 1)
                    self._running = False
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
                    logger.debug("Tool %s: sandbox decision=%s", function_name, decision.value)

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
                    if self._audit:
                        self._audit.tool_start(tool_call_id, function_name, arguments)
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

                    args_by_id = {tc_id: args for tc_id, _, args in tool_tasks}
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
                        if self._audit:
                            self._audit.tool_end(
                                tool_call_id,
                                function_name,
                                success=result.success,
                                result_summary=content if result.success else (result.error or ""),
                                error=result.error,
                                token_usage=self.token_usage.model_dump(),
                                arguments=args_by_id.get(tool_call_id),
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
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step + 1)
                    await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

            logger.warning("Agent hit max steps (%d)", self.max_steps)
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

    async def execute_plan(
        self, cancel_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute tool calls from a plan proposal and continue the agent loop.

        Call after run_stream(plan_mode=True) yielded a PlanProposal.
        The last message in history should be the assistant message with tool_calls.
        """
        # Find the pending tool calls from the last assistant message
        last_msg = self.messages[-1] if self.messages else None
        if not last_msg or last_msg.role != "assistant" or not last_msg.tool_calls:
            yield AgentError(error="No pending plan to execute.", steps=0)
            return

        # Execute the pending tools, then continue with normal run_stream
        async for event in self._execute_tool_calls(last_msg.tool_calls):
            yield event

        # Continue the normal agent loop for remaining steps
        async for event in self.run_stream(cancel_event=cancel_event):
            yield event

    async def reject_plan(self) -> None:
        """Reject a pending plan by adding denial results for all tool calls.

        The agent can then be given new instructions via add_user_message().
        """
        last_msg = self.messages[-1] if self.messages else None
        if not last_msg or last_msg.role != "assistant" or not last_msg.tool_calls:
            return

        for tc in last_msg.tool_calls:
            self.messages.append(
                Message(
                    role="tool",
                    content="Error: Plan rejected by user.",
                    tool_call_id=tc.id,
                    name=tc.function.name,
                )
            )

    async def _execute_tool_calls(
        self, tool_calls: list
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a list of tool calls and append results to history."""
        tool_tasks: list[tuple[str, str, dict]] = []

        for tc in tool_calls:
            tc_id = tc.id
            fn_name = tc.function.name
            args = tc.function.arguments

            decision = self.sandbox.check(fn_name, args)

            if decision == Decision.DENY:
                error_msg = f"Tool blocked by sandbox ({self.sandbox.mode.value} mode): {fn_name}"
                yield ToolEnd(tool_call_id=tc_id, tool_name=fn_name, success=False, content="", error=error_msg)
                self.messages.append(Message(role="tool", content=f"Error: {error_msg}", tool_call_id=tc_id, name=fn_name))
                continue

            if decision == Decision.ASK and self._permission_callback is not None:
                yield PermissionRequest(tool_call_id=tc_id, tool_name=fn_name, arguments=args)
                granted = await self._permission_callback(fn_name, args)
                if not granted:
                    error_msg = f"Permission denied for tool: {fn_name}"
                    yield ToolEnd(tool_call_id=tc_id, tool_name=fn_name, success=False, content="", error=error_msg)
                    self.messages.append(Message(role="tool", content=f"Error: {error_msg}", tool_call_id=tc_id, name=fn_name))
                    continue

            yield ToolStart(tool_call_id=tc_id, tool_name=fn_name, arguments=args)
            if self._audit:
                self._audit.tool_start(tc_id, fn_name, args)
            tool_tasks.append((tc_id, fn_name, args))

        async def _exec(tc_id: str, fn_name: str, args: dict) -> tuple[str, str, ToolResult]:
            if fn_name not in self.tools:
                return tc_id, fn_name, ToolResult(success=False, content="", error=f"Unknown tool: {fn_name}")
            try:
                return tc_id, fn_name, await self.tools[fn_name].execute(**args)
            except Exception as e:
                return tc_id, fn_name, ToolResult(success=False, content="", error=f"Tool execution failed: {e}")

        if tool_tasks:
            args_by_id = {tc_id: args for tc_id, _, args in tool_tasks}
            results = await asyncio.gather(*(_exec(tc_id, fn, args) for tc_id, fn, args in tool_tasks))
            for tc_id, fn_name, result in results:
                content = result.content if result.success else f"Error: {result.error}"
                if self.tool_result_store and result.success:
                    content = self.tool_result_store.store_if_large(content, tc_id)
                yield ToolEnd(tool_call_id=tc_id, tool_name=fn_name, success=result.success, content=content if result.success else "", error=result.error)
                if self._audit:
                    self._audit.tool_end(
                        tc_id,
                        fn_name,
                        success=result.success,
                        result_summary=content if result.success else (result.error or ""),
                        error=result.error,
                        token_usage=self.token_usage.model_dump(),
                        arguments=args_by_id.get(tc_id),
                    )
                self.messages.append(Message(role="tool", content=content, tool_call_id=tc_id, name=fn_name))

    async def _emit_session_end(self, event: AgentDone | AgentCancelled | AgentError) -> None:
        """Emit SESSION_END hook and close the audit log."""
        if self._audit:
            self._audit.close()
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

    def fork(self, rewind: int = 0) -> "Agent":
        """Create an independent fork of this agent with shared history prefix.

        Args:
            rewind: Number of user/assistant turn pairs to rewind before forking.
                0 means fork from current point, 1 means drop the last exchange, etc.

        Returns:
            A new Agent with a copy of the message history up to the fork point.
        """
        # Determine how many messages to keep
        messages = self.messages.copy()
        if rewind > 0:
            # Count turn pairs (user+assistant) from the end and trim
            pairs_removed = 0
            while pairs_removed < rewind and len(messages) > 1:
                # Remove from the end: tool results, assistant, user messages
                if messages[-1].role in ("tool", "assistant"):
                    messages.pop()
                elif messages[-1].role == "user":
                    messages.pop()
                    pairs_removed += 1
                else:
                    break

        forked = Agent(
            llm_client=self.llm,
            system_prompt="",  # will be overwritten below
            tools=list(self.tools.values()),
            max_steps=self.max_steps,
            tool_result_store=self.tool_result_store,
            compact_threshold=self.compact_threshold,
            context_window=self.context_window,
            compact_threshold_pct=self.compact_threshold_pct,
            compaction_reserve=self.compaction_reserve,
            permission_callback=self._permission_callback,
            sandbox=self.sandbox,
            hooks=self.hooks,
            session_id=self.session_id,
        )
        forked.system_prompt = self.system_prompt
        forked.messages = messages
        return forked

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
