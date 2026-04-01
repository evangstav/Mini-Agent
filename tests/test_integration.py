"""Integration test cases - require API key, skipped by default."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool
from mini_agent.tools.mcp_loader import load_mcp_tools_async


@pytest.mark.asyncio
async def test_basic_agent_usage():
    """Test basic agent usage with file creation task.

    Requires a valid API key — skipped if not configured.
    Set MINIMAX_API_KEY env var to run.
    """
    import os

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        pytest.skip("MINIMAX_API_KEY not set")

    with tempfile.TemporaryDirectory() as workspace_dir:
        system_prompt = "You are a helpful AI assistant."

        llm_client = LLMClient(api_key=api_key, model="MiniMax-M2.5")

        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            EditTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=10,
        )

        task = "Create a file named hello.py that prints 'Hello, Mini Agent!', then execute it."
        agent.add_user_message(task)
        result = await agent.run()

        hello_file = Path(workspace_dir) / "hello.py"
        assert hello_file.exists() or "complete" in result.lower()
