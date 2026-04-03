"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .audit import AuditLogger
from .context import SystemPromptBuilder, ToolResultStore, estimate_tokens
from .context_budget import ContextBudget
from .events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    AgentEvent,
    PlanProposal,
    TextChunk,
    ThinkingChunk,
)
from .hooks import HookEvent, HookRegistry, SessionEndPayload, SessionStartPayload
from .llm import LLMClient
from .message_log import MessageLog
from .sandbox import PermissionMode, Sandbox
from .schema import Message, TokenUsage
from .session_memory import SessionMemory
from .tool_execution import ToolExecutor
from .tools.base import Tool

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
        max_steps: int = 100,
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
        auto_end_session: bool = True,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.tool_result_store = tool_result_store

        # Build system prompt with CLAUDE.md and git info if project_dir given
        if project_dir:
            builder = SystemPromptBuilder(system_prompt, project_dir)
            system_prompt = builder.build()

        self.system_prompt = system_prompt
        self.log = MessageLog(system_prompt)
        self._permission_callback = permission_callback
        self.sandbox = sandbox or Sandbox(PermissionMode.AUTO)
        self.hooks = hooks or HookRegistry()
        self.session_id = session_id
        self._running = False
        self._auto_end_session = auto_end_session
        self._session_started = False
        self._session_closed = False
        self.token_usage = TokenUsage()
        self._audit = AuditLogger(session_id) if session_id else None
        self._session_memory = SessionMemory(
            output_path=Path(project_dir) / ".runtime" / "session_memory.md"
        ) if project_dir else SessionMemory()

        # Extracted concerns
        self._budget = ContextBudget(
            context_window=context_window,
            compact_threshold_pct=compact_threshold_pct,
            compaction_reserve=compaction_reserve,
            compact_threshold=compact_threshold,
        )
        self._executor = ToolExecutor(
            tools=self.tools,
            sandbox=self.sandbox,
            permission_callback=permission_callback,
            tool_result_store=tool_result_store,
            audit=self._audit,
            session_memory=self._session_memory,
        )

    # Backward compatibility: self.messages as a property
    @property
    def messages(self) -> list[Message]:
        return self.log.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self.log.replace_prefix(value)

    # Keep compact_threshold accessible for tests/backward compat
    @property
    def compact_threshold(self) -> int:
        return self._budget.threshold

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.log.append_user(content)

    async def run_stream(
        self,
        cancel_event: Optional[asyncio.Event] = None,
        plan_mode: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent loop, yielding events as they occur."""
        if self._running:
            raise RuntimeError("run_stream() is already running; concurrent calls are not supported")
        self._running = True

        try:
            await self._ensure_session_started()
            logger.info("Agent run started (max_steps=%d, session=%s)", self.max_steps, self.session_id)
            _edited_files = False
            _ran_tests = False

            for step in range(self.max_steps):
                if cancel_event and cancel_event.is_set():
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step)
                    if self._auto_end_session:
                        await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

                # Compact context if needed
                await self._budget.maybe_compact(self.log, self.llm, self.hooks)

                # Phase transition nudge: if we're 30% through budget without editing, push to action
                explore_budget = max(10, self.max_steps * 3 // 10)
                if step == explore_budget and not _edited_files:
                    logger.info("Phase nudge at step %d: no edits yet, pushing to action", step)
                    self.log.append_user(
                        f"You've spent {step} steps exploring without making any edits. "
                        f"You MUST now localize the issue and start implementing a fix. "
                        f"If you're unsure of the exact fix, make your best attempt — "
                        f"a wrong fix that can be iterated on is better than no fix at all."
                    )

                # Last-resort nudge: if 80% through budget without editing, make a desperate attempt
                last_resort = self.max_steps * 4 // 5
                if step == last_resort and not _edited_files:
                    logger.warning("Last-resort nudge at step %d: still no edits", step)
                    self.log.append_user(
                        f"CRITICAL: You have only {self.max_steps - step} steps left and haven't made any edits. "
                        f"Write your best-guess fix NOW, even if you're not confident. An imperfect fix is "
                        f"better than no fix."
                    )

                # Think: call LLM with streaming
                logger.debug("Step %d: calling LLM", step)
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
                        error_event = AgentError(error="LLM stream ended without message_complete", steps=step)
                        if self._auto_end_session:
                            await self._emit_session_end(error_event)
                        yield error_event
                        return
                except Exception as e:
                    logger.error("LLM call failed at step %d: %s", step, e)
                    error_event = AgentError(error=f"LLM call failed: {e}", steps=step)
                    if self._auto_end_session:
                        await self._emit_session_end(error_event)
                    yield error_event
                    return

                # Accumulate token usage
                if response.usage:
                    self.token_usage.prompt_tokens += response.usage.prompt_tokens
                    self.token_usage.completion_tokens += response.usage.completion_tokens
                    self.token_usage.total_tokens += response.usage.total_tokens

                # Record assistant message
                self.log.append_assistant(
                    content=response.content,
                    thinking=response.thinking,
                    thinking_signature=getattr(response, "thinking_signature", None),
                    tool_calls=response.tool_calls,
                )

                # No tool calls → task complete (with verification gate)
                if not response.tool_calls:
                    # If agent edited files but never ran tests, nudge it to verify
                    if _edited_files and not _ran_tests and step < self.max_steps - 1:
                        logger.info("Verification gate: agent edited files but didn't run tests, nudging")
                        self.log.append_user(
                            "You edited code but haven't run any tests to verify your changes. "
                            "Please run the relevant tests before finishing."
                        )
                        _ran_tests = True  # Don't nudge twice
                        continue

                    logger.info("Agent completed at step %d", step + 1)
                    done_event = AgentDone(content=response.content, steps=step + 1)
                    if self._auto_end_session:
                        await self._emit_session_end(done_event)
                    yield done_event
                    return

                # Plan mode: yield proposal instead of executing tools
                if plan_mode:
                    proposed = [
                        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in response.tool_calls
                    ]
                    yield PlanProposal(proposed_calls=proposed, steps=step + 1)
                    self._running = False
                    return

                # Track edits and test runs for the verification gate
                for tc in response.tool_calls:
                    fn = tc.function.name
                    if fn in ("edit_file", "write_file"):
                        _edited_files = True
                    if fn == "bash":
                        cmd = tc.function.arguments.get("command", "")
                        if any(kw in cmd for kw in ("pytest", "python -m pytest", "test", "npm test", "go test", "cargo test")):
                            _ran_tests = True

                # Act: execute tool calls, Observe: collect results
                async for event in self._executor.execute_batch(response.tool_calls, self.token_usage):
                    yield event

                # Add step counter to last result message so the LLM knows its budget
                result_msgs = self._executor.result_messages
                if result_msgs:
                    last = result_msgs[-1]
                    remaining = self.max_steps - step - 1
                    step_info = f"\n[Step {step + 1}/{self.max_steps} — {remaining} steps remaining]"
                    content = last.content if isinstance(last.content, str) else str(last.content)
                    from .schema import Message
                    result_msgs[-1] = Message(
                        role=last.role, content=content + step_info,
                        tool_call_id=last.tool_call_id, name=last.name,
                    )
                self.log.extend(result_msgs)

                # Write session memory snapshot if due
                token_est = self.log.estimate_tokens()
                if self._session_memory.should_update(token_est):
                    self._session_memory.write_snapshot(token_est)

                if cancel_event and cancel_event.is_set():
                    cancel_event_obj = AgentCancelled(content="Task cancelled by user.", steps=step + 1)
                    if self._auto_end_session:
                        await self._emit_session_end(cancel_event_obj)
                    yield cancel_event_obj
                    return

            logger.warning("Agent hit max steps (%d)", self.max_steps)
            error_event = AgentError(
                error=f"Task couldn't be completed after {self.max_steps} steps.",
                steps=self.max_steps,
            )
            if self._auto_end_session:
                await self._emit_session_end(error_event)
            yield error_event
        finally:
            self._running = False

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached."""
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
        """Execute tool calls from a plan proposal and continue the agent loop."""
        last_msg = self.messages[-1] if self.messages else None
        if not last_msg or last_msg.role != "assistant" or not last_msg.tool_calls:
            yield AgentError(error="No pending plan to execute.", steps=0)
            return

        # Execute the pending tools
        async for event in self._executor.execute_batch(last_msg.tool_calls, self.token_usage):
            yield event
        self.log.extend(self._executor.result_messages)

        # Continue the normal agent loop
        async for event in self.run_stream(cancel_event=cancel_event):
            yield event

    async def reject_plan(self) -> None:
        """Reject a pending plan by adding denial results for all tool calls."""
        last_msg = self.messages[-1] if self.messages else None
        if not last_msg or last_msg.role != "assistant" or not last_msg.tool_calls:
            return
        for tc in last_msg.tool_calls:
            self.log.append_tool_result(
                content="Error: Plan rejected by user.",
                tool_call_id=tc.id,
                name=tc.function.name,
            )

    async def _emit_session_end(self, event: AgentDone | AgentCancelled | AgentError) -> None:
        """Emit SESSION_END hook and close the audit log once."""
        if not self._session_started or self._session_closed:
            return
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
                messages=self.log.copy(),
                final_event_type=event_type,
                steps=event.steps,
            ),
        )
        self._session_closed = True

    async def _ensure_session_started(self) -> None:
        """Emit SESSION_START exactly once."""
        if self._session_started and not self._session_closed:
            return
        await self.hooks.emit(
            HookEvent.SESSION_START,
            SessionStartPayload(
                session_id=self.session_id,
                system_prompt=self.system_prompt,
            ),
        )
        self._session_started = True
        self._session_closed = False

    def fork(self, rewind: int = 0) -> "Agent":
        """Create an independent fork with shared history prefix."""
        forked_log = self.log.fork(rewind=rewind)

        forked = Agent(
            llm_client=self.llm,
            system_prompt="",  # overwritten below
            tools=list(self.tools.values()),
            max_steps=self.max_steps,
            tool_result_store=self.tool_result_store,
            compact_threshold=self._budget.threshold,
            permission_callback=self._permission_callback,
            sandbox=self.sandbox,
            hooks=self.hooks,
            session_id=self.session_id,
            auto_end_session=self._auto_end_session,
        )
        forked.system_prompt = self.system_prompt
        forked.log = forked_log
        forked._session_started = self._session_started
        forked._session_closed = self._session_closed
        return forked

    async def end_session(self) -> None:
        """Clean up session resources."""
        token_est = self.log.estimate_tokens()
        self._session_memory.update_state("Session ended")
        self._session_memory.write_snapshot(token_est)
        await self._emit_session_end(
            AgentDone(content="Session ended by user.", steps=0)
        )

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.log.copy()
