"""Core Agent implementation — generator-based event-driven loop."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

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
        permission_callback: object = None,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
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
                        content=result.content if result.success else f"Error: {result.error}",
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

        Args:
            cancel_event: Optional event to cancel execution.

        Returns:
            The final response content.
        """
        final_content = ""
        async for event in self.run_stream(cancel_event=cancel_event):
            if isinstance(event, AgentDone):
                return event.content
            elif isinstance(event, AgentError):
                return event.error
        return final_content

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
