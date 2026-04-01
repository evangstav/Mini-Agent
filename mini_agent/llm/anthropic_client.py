"""Anthropic LLM client implementation."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, StreamDelta, TokenUsage, ToolCall
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClientBase):
    """LLM client using Anthropic's protocol.

    This client uses the official Anthropic SDK and supports:
    - Extended thinking content
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2.7",
        retry_config: RetryConfig | None = None,
        max_tokens: int = 16384,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax Anthropic endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
            max_tokens: Maximum tokens in response (default: 16384)
        """
        super().__init__(api_key, api_base, model, retry_config)
        self.max_tokens = max_tokens

        # Initialize Anthropic async client
        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    async def _make_api_request(
        self,
        system_message: str | None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> anthropic.types.Message:
        """Execute API request (core method that can be retried).

        Args:
            system_message: Optional system message
            api_messages: List of messages in Anthropic format
            tools: Optional list of tools

        Returns:
            Anthropic Message response

        Raises:
            Exception: API call failed
        """
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self._convert_tools(tools)

        # Use Anthropic SDK's async messages.create
        response = await self.client.messages.create(**params)
        return response

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in Anthropic dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "to_schema"):
                # Tool object with to_schema method
                result.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            # For user and assistant messages
            if msg.role in ["user", "assistant"]:
                # Handle assistant messages with thinking or tool calls
                if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    # Build content blocks for assistant with thinking and/or tool calls
                    content_blocks = []

                    # Add thinking block if present
                    if msg.thinking:
                        content_blocks.append(
                            {"type": "thinking", "text": msg.thinking}
                        )

                    # Add text content if present
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})

                    # Add tool use blocks
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,
                                }
                            )

                    api_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    api_messages.append({"role": msg.role, "content": msg.content})

            # For tool result messages — batch consecutive tool results into one user message
            elif msg.role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                # If the previous message is already a batched tool_result user message, append
                if (api_messages
                        and api_messages[-1]["role"] == "user"
                        and isinstance(api_messages[-1]["content"], list)
                        and api_messages[-1]["content"]
                        and api_messages[-1]["content"][0].get("type") == "tool_result"):
                    api_messages[-1]["content"].append(tool_result_block)
                else:
                    api_messages.append(
                        {
                            "role": "user",
                            "content": [tool_result_block],
                        }
                    )

        return system_message, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        system_message, api_messages = self._convert_messages(messages)

        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: Anthropic Message response

        Returns:
            LLMResponse object
        """
        # Extract text content, thinking, and tool calls
        text_content = ""
        thinking_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "tool_use":
                # Parse Anthropic tool_use block
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )

        # Extract token usage from response
        # Anthropic usage includes: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
        usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0
            cache_read_tokens = (
                getattr(response.usage, "cache_read_input_tokens", 0) or 0
            )
            cache_creation_tokens = (
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            total_input_tokens = (
                input_tokens + cache_read_tokens + cache_creation_tokens
            )
            usage = TokenUsage(
                prompt_tokens=total_input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_input_tokens + output_tokens,
            )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from Anthropic LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse containing the generated content
        """
        # Prepare request
        request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(
                config=self.retry_config, on_retry=self.retry_callback
            )
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )

        # Parse and return response
        return self._parse_response(response)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream response from Anthropic LLM, yielding deltas as tokens arrive.

        Uses client.messages.stream() for real-time token delivery.
        Token usage and tool calls are accumulated and delivered in the
        final message_complete event.
        """
        request_params = self._prepare_request(messages, tools)

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": request_params["api_messages"],
        }
        if request_params["system_message"]:
            params["system"] = request_params["system_message"]
        if request_params["tools"]:
            params["tools"] = self._convert_tools(request_params["tools"])

        # Accumulate full content for the final LLMResponse
        text_content = ""
        thinking_content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None
        input_json_buf = ""

        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        input_json_buf = ""

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_content += delta.text
                        yield StreamDelta(type="text_delta", text=delta.text)
                    elif delta.type == "thinking_delta":
                        thinking_content += delta.thinking
                        yield StreamDelta(type="thinking_delta", text=delta.thinking)
                    elif delta.type == "input_json_delta":
                        input_json_buf += delta.partial_json

                elif event_type == "content_block_stop":
                    if current_tool is not None:
                        import json as _json
                        arguments = _json.loads(input_json_buf) if input_json_buf else {}
                        tool_calls.append(
                            ToolCall(
                                id=current_tool["id"],
                                type="function",
                                function=FunctionCall(
                                    name=current_tool["name"],
                                    arguments=arguments,
                                ),
                            )
                        )
                        current_tool = None
                        input_json_buf = ""

            # After stream completes, get the final message for usage info
            final_message = await stream.get_final_message()

        usage = None
        if hasattr(final_message, "usage") and final_message.usage:
            input_tokens = final_message.usage.input_tokens or 0
            output_tokens = final_message.usage.output_tokens or 0
            cache_read = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0
            total_input = input_tokens + cache_read + cache_create
            usage = TokenUsage(
                prompt_tokens=total_input,
                completion_tokens=output_tokens,
                total_tokens=total_input + output_tokens,
            )

        yield StreamDelta(
            type="message_complete",
            response=LLMResponse(
                content=text_content,
                thinking=thinking_content if thinking_content else None,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=final_message.stop_reason or "stop",
                usage=usage,
            ),
        )
