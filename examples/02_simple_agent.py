"""Example 2: Simple Agent Usage

This example demonstrates how to create and run a basic agent
to perform simple file operations.

Set MINIMAX_API_KEY env var before running.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool


async def demo_file_creation():
    """Demo: Agent creates a file based on user request."""
    print("\n" + "=" * 60)
    print("Demo: Agent-Driven File Creation")
    print("=" * 60)

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Set MINIMAX_API_KEY env var first.")
        return

    with tempfile.TemporaryDirectory() as workspace_dir:
        llm_client = LLMClient(api_key=api_key)

        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            EditTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        agent = Agent(
            llm_client=llm_client,
            system_prompt="You are a helpful AI assistant that can use tools.",
            tools=tools,
            max_steps=10,
        )

        task = "Create a Python file named 'hello.py' that prints 'Hello, Mini Agent!'"
        agent.add_user_message(task)
        result = await agent.run()
        print(f"\nResult: {result}")


async def main():
    await demo_file_creation()


if __name__ == "__main__":
    asyncio.run(main())
