"""Demo: Using Tool schemas with base Tool class.

Set MINIMAX_API_KEY env var before running.
"""

import asyncio
import os
from typing import Any

from mini_agent import LLMClient, LLMProvider
from mini_agent.schema import Message
from mini_agent.tools.base import Tool, ToolResult


class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get current weather for a location."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="72F, Sunny")


async def main():
    tool = WeatherTool()
    print("Anthropic schema:", tool.to_schema())
    print("OpenAI schema:", tool.to_openai_schema())

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Set MINIMAX_API_KEY to test with LLM.")
        return

    client = LLMClient(api_key=api_key, provider=LLMProvider.ANTHROPIC)
    messages = [Message(role="user", content="What's the weather in Tokyo?")]
    response = await client.generate(messages, tools=[tool])

    print(f"Response: {response.content}")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")


if __name__ == "__main__":
    asyncio.run(main())
