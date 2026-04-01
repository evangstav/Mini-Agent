"""Test cases for Bash Tool."""

import pytest

from mini_agent.tools.bash_tool import BashTool


@pytest.mark.asyncio
async def test_foreground_command():
    """Test executing a simple foreground command."""
    bash_tool = BashTool()
    result = await bash_tool.execute(command="echo 'Hello from foreground'")

    assert result.success
    assert "Hello from foreground" in result.content


@pytest.mark.asyncio
async def test_foreground_command_with_stderr():
    """Test command that outputs to both stdout and stderr."""
    bash_tool = BashTool()
    result = await bash_tool.execute(command="echo 'stdout message' && echo 'stderr message' >&2")

    assert result.success
    assert "stdout message" in result.content
    assert "stderr message" in result.content


@pytest.mark.asyncio
async def test_command_failure():
    """Test command that fails with non-zero exit code."""
    bash_tool = BashTool()
    result = await bash_tool.execute(command="ls /nonexistent_directory_12345")

    assert not result.success
    assert result.error is not None


@pytest.mark.asyncio
async def test_command_timeout():
    """Test command timeout."""
    bash_tool = BashTool()
    result = await bash_tool.execute(command="sleep 10", timeout=1)

    assert not result.success
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_timeout_validation():
    """Test timeout parameter validation."""
    bash_tool = BashTool()

    # Test with timeout > 600 (should be capped to 600)
    result = await bash_tool.execute(command="echo 'test'", timeout=1000)
    assert result.success

    # Test with timeout < 1 (should be set to 1)
    result = await bash_tool.execute(command="echo 'test'", timeout=0)
    assert result.success
