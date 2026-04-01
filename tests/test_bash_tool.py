"""Test cases for Bash Tool."""

import pytest

from mini_agent.tools.bash_tool import (
    BashKillTool,
    BashOutputTool,
    BashTool,
    _background_processes,
)


@pytest.fixture(autouse=True)
def _clean_bg_processes():
    """Ensure background process registry is clean between tests."""
    _background_processes.clear()
    yield
    # Kill any leftover processes
    for bg in list(_background_processes.values()):
        if bg.is_running:
            bg.process.kill()
    _background_processes.clear()


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


# --- Background execution tests ---


@pytest.mark.asyncio
async def test_background_returns_process_id():
    """Test that run_in_background returns a process_id."""
    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="sleep 30", run_in_background=True
    )

    assert result.success
    assert "process_id=" in result.content
    assert "bash_output" in result.content
    assert len(_background_processes) == 1


@pytest.mark.asyncio
async def test_bash_output_reads_new_output():
    """Test reading output from a background process."""
    import asyncio

    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="echo 'bg hello' && sleep 30", run_in_background=True
    )
    # Extract process_id from result
    pid = int(result.content.split("process_id=")[1].split(")")[0])

    # Give it a moment to produce output
    await asyncio.sleep(0.3)

    output_tool = BashOutputTool()
    out = await output_tool.execute(process_id=pid)

    assert out.success
    assert "bg hello" in out.content
    assert "running" in out.content


@pytest.mark.asyncio
async def test_bash_output_invalid_id():
    """Test bash_output with invalid process ID."""
    output_tool = BashOutputTool()
    out = await output_tool.execute(process_id=9999)

    assert not out.success
    assert "No background process" in out.error


@pytest.mark.asyncio
async def test_bash_kill_stops_process():
    """Test killing a background process."""
    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="sleep 300", run_in_background=True
    )
    pid = int(result.content.split("process_id=")[1].split(")")[0])

    kill_tool = BashKillTool()
    kill_result = await kill_tool.execute(process_id=pid)

    assert kill_result.success
    assert "stopped" in kill_result.content.lower() or "Process" in kill_result.content
    assert pid not in _background_processes


@pytest.mark.asyncio
async def test_bash_kill_invalid_id():
    """Test bash_kill with invalid process ID."""
    kill_tool = BashKillTool()
    result = await kill_tool.execute(process_id=9999)

    assert not result.success
    assert "No background process" in result.error


@pytest.mark.asyncio
async def test_background_process_completes():
    """Test reading output after background process finishes."""
    import asyncio

    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="echo 'done quickly'", run_in_background=True
    )
    pid = int(result.content.split("process_id=")[1].split(")")[0])

    # Wait for process to finish
    await asyncio.sleep(0.5)

    output_tool = BashOutputTool()
    out = await output_tool.execute(process_id=pid)

    assert out.success
    assert "done quickly" in out.content
    assert "exited" in out.content


@pytest.mark.asyncio
async def test_background_blocked_command():
    """Background mode still enforces safety checks."""
    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="rm -rf /", run_in_background=True
    )

    assert not result.success
    assert len(_background_processes) == 0


@pytest.mark.asyncio
async def test_multiple_background_processes():
    """Test running multiple background processes."""
    bash_tool = BashTool()
    r1 = await bash_tool.execute(command="sleep 30", run_in_background=True)
    r2 = await bash_tool.execute(command="sleep 30", run_in_background=True)

    pid1 = int(r1.content.split("process_id=")[1].split(")")[0])
    pid2 = int(r2.content.split("process_id=")[1].split(")")[0])

    assert pid1 != pid2
    assert len(_background_processes) == 2
