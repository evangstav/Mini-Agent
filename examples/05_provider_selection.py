"""Example: Using LLMClient with different providers.

Set MINIMAX_API_KEY env var before running.
"""

import asyncio
import os

from mini_agent import LLMClient, LLMProvider, Message


async def main():
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Set MINIMAX_API_KEY env var first.")
        return

    messages = [Message(role="user", content="Say 'Hello' and nothing else.")]

    # Anthropic provider (default)
    client = LLMClient(api_key=api_key, provider=LLMProvider.ANTHROPIC)
    response = await client.generate(messages)
    print(f"Anthropic: {response.content}")

    # OpenAI provider
    client = LLMClient(api_key=api_key, provider=LLMProvider.OPENAI)
    response = await client.generate(messages)
    print(f"OpenAI: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
