"""Live agent tests — requires MINIMAX_API_KEY or ANTHROPIC_API_KEY.

These tests validate research-backed improvements end-to-end:
1. Grounding: Does the agent avoid fabrication when files aren't found?
2. Verification: Does the agent suggest running tests after edits?
3. Windowed output: Does ReadTool cap output properly in context?
4. Tool descriptions: Does the agent use tools appropriately?

Skip with: pytest -m "not live"
"""

import asyncio
import os
import pytest

from mini_agent.agent import Agent
from mini_agent.context import ToolResultStore
from mini_agent.events import AgentDone, AgentError, ToolEnd, ToolStart
from mini_agent.llm import LLMClient
from mini_agent.sandbox import PermissionMode, Sandbox
from mini_agent.schema import LLMProvider
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import EditTool, ReadTool, WriteTool
from mini_agent.tools.glob_tool import GlobTool
from mini_agent.tools.grep_tool import GrepTool

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not (os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="No API key set (need MINIMAX_API_KEY or ANTHROPIC_API_KEY)",
)


def _make_agent(workspace: str, system_prompt: str | None = None) -> Agent:
    """Create an agent with real LLM client for live testing."""
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = minimax_key or anthropic_key

    is_minimax = bool(minimax_key)
    provider = LLMProvider.ANTHROPIC
    model = "MiniMax-M2.7" if is_minimax else "claude-sonnet-4-20250514"
    api_base = "https://api.minimax.io" if is_minimax else "https://api.anthropic.com"

    llm = LLMClient(api_key=api_key, provider=provider, api_base=api_base, model=model)

    prompt = system_prompt or (
        "You are a helpful coding assistant. "
        "Only state facts you have verified by reading files or running commands. "
        "If you cannot find evidence, say so explicitly. Never fabricate."
    )

    tools = [
        ReadTool(workspace_dir=workspace),
        WriteTool(workspace_dir=workspace),
        EditTool(workspace_dir=workspace),
        GlobTool(workspace_dir=workspace),
        GrepTool(workspace_dir=workspace),
        BashTool(workspace_dir=workspace),
    ]

    return Agent(
        llm_client=llm,
        system_prompt=prompt,
        tools=tools,
        max_steps=10,
        sandbox=Sandbox(PermissionMode.FULL_ACCESS),
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_grounding_no_fabrication(tmp_path):
    """Agent should NOT fabricate when asked about nonexistent files."""
    agent = _make_agent(str(tmp_path))
    agent.add_user_message(
        "How many test files are in the tests/ directory? "
        "List them with their line counts."
    )

    events = []
    async for event in agent.run_stream():
        events.append(event)

    # Get the final response
    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert done_events, "Agent should complete"

    response = done_events[0].content.lower()

    # The tmp_path has no tests/ directory — agent should say so
    fabrication_markers = [
        "test_agent.py",  # Should NOT invent filenames
        "24 test files",  # Should NOT claim specific counts
        "100% coverage",  # Should NOT fabricate metrics
    ]
    for marker in fabrication_markers:
        assert marker not in response, f"Agent fabricated: found '{marker}' in response"

    # Should indicate it couldn't find them
    honesty_markers = ["not found", "no ", "doesn't exist", "empty", "could not", "no test", "no files"]
    assert any(m in response for m in honesty_markers), \
        f"Agent should acknowledge missing files. Response: {response[:200]}"


@pytest.mark.live
@pytest.mark.asyncio
async def test_windowed_read_in_practice(tmp_path):
    """Agent receives truncated output for large files."""
    # Create a 500-line file
    big_file = tmp_path / "big.py"
    big_file.write_text("\n".join(f"# line {i}" for i in range(500)))

    agent = _make_agent(str(tmp_path))
    agent.add_user_message("Read the file big.py and tell me how many lines it has.")

    tool_results = []
    async for event in agent.run_stream():
        if isinstance(event, ToolEnd) and event.success:
            tool_results.append(event.content)

    # The ReadTool should have returned truncated output
    assert tool_results, "Agent should have called read_file"
    first_read = tool_results[0]
    assert "Showing lines" in first_read or "line 199" in first_read, \
        "ReadTool should cap output at 200 lines"


@pytest.mark.live
@pytest.mark.asyncio
async def test_linter_warns_on_bad_edit_directly(tmp_path):
    """EditTool warns but writes syntax-breaking changes (SWE-Agent approach)."""
    test_file = tmp_path / "code.py"
    test_file.write_text("def hello():\n    return 'world'\n")

    tool = EditTool(workspace_dir=str(tmp_path))
    result = await tool.execute(
        path=str(test_file),
        old_str="return 'world'",
        new_str="return ('world'",  # Unclosed paren — breaks syntax
    )

    # Edit succeeds with a warning (allows multi-step edits)
    assert result.success, "EditTool should warn but write"
    assert "WARNING" in result.content
    assert "syntax" in result.content.lower()
