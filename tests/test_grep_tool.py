"""Tests for GrepTool."""

import pytest

from mini_agent.tools.grep_tool import GrepTool


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_finds_pattern(self, tmp_path):
        (tmp_path / "test.py").write_text("def hello():\n    pass\n")
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="hello")
        assert result.success
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        (tmp_path / "test.py").write_text("def foo():\n    pass\n")
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="nonexistent_pattern_xyz")
        assert result.success
        assert "No matches" in result.content

    @pytest.mark.asyncio
    async def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.py").write_text("Hello World\n")
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="hello", case_insensitive=True)
        assert result.success
        assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_files_with_matches_mode(self, tmp_path):
        (tmp_path / "a.py").write_text("def target():\n")
        (tmp_path / "b.py").write_text("no match here\n")
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="target", output_mode="files_with_matches")
        assert result.success
        assert "a.py" in result.content
        assert "b.py" not in result.content

    @pytest.mark.asyncio
    async def test_path_escapes_workspace(self, tmp_path):
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="test", path="/etc")
        assert not result.success
        assert "escapes workspace" in result.error

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_path):
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="test", path=str(tmp_path / "nonexistent"))
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_name_and_description(self, tmp_path):
        tool = GrepTool(workspace_dir=str(tmp_path))
        assert tool.name == "grep"
        assert "regex" in tool.description.lower() or "search" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_parameters_schema(self, tmp_path):
        tool = GrepTool(workspace_dir=str(tmp_path))
        params = tool.parameters
        assert "pattern" in params["properties"]
        assert "pattern" in params["required"]

    @pytest.mark.asyncio
    async def test_glob_filter(self, tmp_path):
        (tmp_path / "code.py").write_text("target_string\n")
        (tmp_path / "data.txt").write_text("target_string\n")
        tool = GrepTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="target_string", glob="*.py")
        assert result.success
        assert "code.py" in result.content
