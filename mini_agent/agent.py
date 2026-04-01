"""Core Agent implementation — minimal think-act-observe loop."""

import asyncio
import json
from typing import Optional

from .llm import LLMClient
from .schema import Message
from .tools.base import Tool, ToolResult


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        Args:
            cancel_event: Optional event to cancel execution.

        Returns:
            The final response content.
        """
        for step in range(self.max_steps):
            if cancel_event and cancel_event.is_set():
                return "Task cancelled by user."

            # Think: call LLM
            try:
                response = await self.llm.generate(
                    messages=self.messages, tools=list(self.tools.values())
                )
            except Exception as e:
                return f"LLM call failed: {e}"

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
                return response.content

            # Act: execute tool calls, Observe: collect results
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

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

                self.messages.append(
                    Message(
                        role="tool",
                        content=result.content if result.success else f"Error: {result.error}",
                        tool_call_id=tool_call_id,
                        name=function_name,
                    )
                )

                if cancel_event and cancel_event.is_set():
                    return "Task cancelled by user."

        return f"Task couldn't be completed after {self.max_steps} steps."

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
