"""Test cases for git tools."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.git_tool import (
    GitBranchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )
    # Initial commit
    readme = tmp_path / "README.md"
    readme.write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )
    return tmp_path


@pytest.mark.asyncio
async def test_git_status_clean(git_repo):
    """git_status on a clean repo."""
    tool = GitStatusTool(workspace_dir=str(git_repo))
    result = await tool.execute()
    assert result.success
    assert "Working tree clean" in result.content


@pytest.mark.asyncio
async def test_git_status_dirty(git_repo):
    """git_status detects uncommitted changes."""
    (git_repo / "new.txt").write_text("hello")
    tool = GitStatusTool(workspace_dir=str(git_repo))
    result = await tool.execute()
    assert result.success
    assert "new.txt" in result.content


@pytest.mark.asyncio
async def test_git_status_not_a_repo(tmp_path):
    """git_status fails gracefully outside a repo."""
    tool = GitStatusTool(workspace_dir=str(tmp_path))
    result = await tool.execute()
    assert not result.success
    assert "Not a git repository" in result.error


@pytest.mark.asyncio
async def test_git_diff_no_changes(git_repo):
    """git_diff with no changes."""
    tool = GitDiffTool(workspace_dir=str(git_repo))
    result = await tool.execute()
    assert result.success
    assert "(no changes)" in result.content


@pytest.mark.asyncio
async def test_git_diff_unstaged(git_repo):
    """git_diff shows unstaged changes."""
    readme = git_repo / "README.md"
    readme.write_text("# Modified\n")
    tool = GitDiffTool(workspace_dir=str(git_repo))
    result = await tool.execute()
    assert result.success
    assert "Modified" in result.content


@pytest.mark.asyncio
async def test_git_diff_staged(git_repo):
    """git_diff staged=True shows cached changes."""
    readme = git_repo / "README.md"
    readme.write_text("# Staged\n")
    subprocess.run(["git", "add", "."], cwd=str(git_repo), check=True, capture_output=True)
    tool = GitDiffTool(workspace_dir=str(git_repo))
    result = await tool.execute(staged=True)
    assert result.success
    assert "Staged" in result.content


@pytest.mark.asyncio
async def test_git_diff_stat_only(git_repo):
    """git_diff stat_only shows summary."""
    (git_repo / "a.txt").write_text("content")
    subprocess.run(["git", "add", "."], cwd=str(git_repo), check=True, capture_output=True)
    tool = GitDiffTool(workspace_dir=str(git_repo))
    result = await tool.execute(staged=True, stat_only=True)
    assert result.success
    assert "a.txt" in result.content


@pytest.mark.asyncio
async def test_git_commit(git_repo):
    """git_commit stages and commits files."""
    new_file = git_repo / "feature.py"
    new_file.write_text("print('hello')\n")
    tool = GitCommitTool(workspace_dir=str(git_repo))
    result = await tool.execute(message="feat: add feature", files=["feature.py"])
    assert result.success
    assert "feat: add feature" in result.content


@pytest.mark.asyncio
async def test_git_commit_add_all(git_repo):
    """git_commit with add_all=True."""
    (git_repo / "a.txt").write_text("a")
    (git_repo / "b.txt").write_text("b")
    tool = GitCommitTool(workspace_dir=str(git_repo))
    result = await tool.execute(message="chore: add files", add_all=True)
    assert result.success
    assert "chore: add files" in result.content


@pytest.mark.asyncio
async def test_git_commit_nothing_staged(git_repo):
    """git_commit fails when nothing staged."""
    tool = GitCommitTool(workspace_dir=str(git_repo))
    result = await tool.execute(message="empty commit")
    assert not result.success
    assert "Nothing staged" in result.error


@pytest.mark.asyncio
async def test_git_log(git_repo):
    """git_log shows commit history."""
    tool = GitLogTool(workspace_dir=str(git_repo))
    result = await tool.execute()
    assert result.success
    assert "Initial commit" in result.content


@pytest.mark.asyncio
async def test_git_log_oneline(git_repo):
    """git_log oneline format."""
    tool = GitLogTool(workspace_dir=str(git_repo))
    result = await tool.execute(oneline=True)
    assert result.success
    assert "Initial commit" in result.content


@pytest.mark.asyncio
async def test_git_log_path_filter(git_repo):
    """git_log filtered by path."""
    tool = GitLogTool(workspace_dir=str(git_repo))
    result = await tool.execute(path="README.md")
    assert result.success
    assert "Initial commit" in result.content

    result2 = await tool.execute(path="nonexistent.txt")
    assert result2.success
    assert "(no commits)" in result2.content


@pytest.mark.asyncio
async def test_git_branch_list(git_repo):
    """git_branch list shows branches."""
    tool = GitBranchTool(workspace_dir=str(git_repo))
    result = await tool.execute(action="list")
    assert result.success
    assert "main" in result.content or "master" in result.content


@pytest.mark.asyncio
async def test_git_branch_create(git_repo):
    """git_branch create makes a new branch."""
    tool = GitBranchTool(workspace_dir=str(git_repo))
    result = await tool.execute(action="create", branch_name="feature/test")
    assert result.success
    assert "feature/test" in result.content


@pytest.mark.asyncio
async def test_git_branch_switch(git_repo):
    """git_branch switch changes branch."""
    # Create a branch first
    subprocess.run(
        ["git", "branch", "other"],
        cwd=str(git_repo), check=True, capture_output=True,
    )
    tool = GitBranchTool(workspace_dir=str(git_repo))
    result = await tool.execute(action="switch", branch_name="other")
    assert result.success
    assert "other" in result.content


@pytest.mark.asyncio
async def test_git_branch_create_missing_name(git_repo):
    """git_branch create without name fails."""
    tool = GitBranchTool(workspace_dir=str(git_repo))
    result = await tool.execute(action="create")
    assert not result.success
    assert "branch_name required" in result.error


@pytest.mark.asyncio
async def test_schema_generation():
    """All git tools produce valid schemas."""
    tools = [GitStatusTool(), GitDiffTool(), GitCommitTool(), GitLogTool(), GitBranchTool()]
    for tool in tools:
        schema = tool.to_schema()
        assert schema["name"].startswith("git_")
        assert "description" in schema
        assert "input_schema" in schema
        openai = tool.to_openai_schema()
        assert openai["type"] == "function"
        assert openai["function"]["name"].startswith("git_")
