"""Tests for GlobTool."""

import pytest

from mini_agent.tools.glob_tool import GlobTool


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_finds_python_files(self, tmp_path):
        (tmp_path / "foo.py").write_text("pass")
        (tmp_path / "bar.py").write_text("pass")
        (tmp_path / "readme.txt").write_text("hello")
        tool = GlobTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="*.py")
        assert result.success
        assert "foo.py" in result.content
        assert "bar.py" in result.content
        assert "readme.txt" not in result.content

    @pytest.mark.asyncio
    async def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("pass")
        tool = GlobTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="**/*.py")
        assert result.success
        assert "main.py" in result.content

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        tool = GlobTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="*.xyz")
        assert result.success
        assert "No matches" in result.content

    @pytest.mark.asyncio
    async def test_path_escapes_workspace(self, tmp_path):
        tool = GlobTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="*.py", path="/etc")
        assert not result.success
        assert "escapes workspace" in result.error

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, tmp_path):
        tool = GlobTool(workspace_dir=str(tmp_path))
        result = await tool.execute(pattern="*.py", path=str(tmp_path / "nonexistent"))
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_name_and_description(self, tmp_path):
        tool = GlobTool(workspace_dir=str(tmp_path))
        assert tool.name == "glob"
        assert "pattern" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_parameters_schema(self, tmp_path):
        tool = GlobTool(workspace_dir=str(tmp_path))
        params = tool.parameters
        assert "pattern" in params["properties"]
        assert "pattern" in params["required"]
