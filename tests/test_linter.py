"""Tests for linter-integrated editing (SWE-Agent research-backed feature)."""

import pytest

from mini_agent.tools.file_tools import _lint_file, _check_python_syntax, _check_json_syntax, EditTool, WriteTool
from pathlib import Path


class TestPythonLinter:
    def test_valid_python(self):
        assert _check_python_syntax(Path("test.py"), "x = 1\ndef foo():\n    pass\n") is None

    def test_invalid_python_unclosed_paren(self):
        result = _check_python_syntax(Path("test.py"), "def foo(")
        assert result is not None
        assert "syntax error" in result.lower()

    def test_invalid_python_bad_indent(self):
        result = _check_python_syntax(Path("test.py"), "def foo():\nx = 1")
        assert result is not None

    def test_empty_python(self):
        assert _check_python_syntax(Path("test.py"), "") is None

    def test_python_with_comments(self):
        assert _check_python_syntax(Path("test.py"), "# comment\nx = 1") is None


class TestJsonLinter:
    def test_valid_json(self):
        assert _check_json_syntax(Path("test.json"), '{"key": "value"}') is None

    def test_invalid_json(self):
        result = _check_json_syntax(Path("test.json"), '{key: value}')
        assert result is not None
        assert "syntax error" in result.lower()

    def test_empty_json_object(self):
        assert _check_json_syntax(Path("test.json"), "{}") is None

    def test_json_array(self):
        assert _check_json_syntax(Path("test.json"), "[1, 2, 3]") is None


class TestLintFile:
    def test_python_file(self):
        assert _lint_file(Path("test.py"), "x = 1") is None

    def test_json_file(self):
        assert _lint_file(Path("test.json"), '{"a": 1}') is None

    def test_unknown_extension(self):
        assert _lint_file(Path("test.md"), "anything") is None

    def test_py_extension_case_insensitive(self):
        # .PY should also be linted (Path.suffix preserves case but we lower it)
        assert _lint_file(Path("test.PY"), "x = 1") is None


class TestEditToolLinting:
    @pytest.mark.asyncio
    async def test_edit_rejects_syntax_breaking_change(self, tmp_path):
        """EditTool should reject edits that break Python syntax."""
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n", encoding="utf-8")

        tool = EditTool(workspace_dir=str(tmp_path))
        result = await tool.execute(
            path=str(f),
            old_str="return 1",
            new_str="return (",  # Breaks syntax
        )

        assert not result.success
        assert "syntax" in result.error.lower()
        # File should be UNCHANGED
        assert f.read_text() == "def foo():\n    return 1\n"

    @pytest.mark.asyncio
    async def test_edit_accepts_valid_change(self, tmp_path):
        """EditTool should accept edits that maintain valid syntax."""
        f = tmp_path / "test.py"
        f.write_text("x = 1\n", encoding="utf-8")

        tool = EditTool(workspace_dir=str(tmp_path))
        result = await tool.execute(
            path=str(f),
            old_str="x = 1",
            new_str="x = 2",
        )

        assert result.success
        assert f.read_text() == "x = 2\n"

    @pytest.mark.asyncio
    async def test_edit_non_python_skips_lint(self, tmp_path):
        """EditTool should not lint non-Python files."""
        f = tmp_path / "readme.md"
        f.write_text("hello world", encoding="utf-8")

        tool = EditTool(workspace_dir=str(tmp_path))
        result = await tool.execute(
            path=str(f),
            old_str="hello",
            new_str="broken {{ syntax",
        )

        assert result.success  # No linting for .md files

    @pytest.mark.asyncio
    async def test_edit_includes_verify_hint_for_code(self, tmp_path):
        """EditTool should suggest running tests for code files."""
        f = tmp_path / "test.py"
        f.write_text("x = 1\n", encoding="utf-8")

        tool = EditTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f), old_str="x = 1", new_str="x = 2")

        assert result.success
        assert "test" in result.content.lower()


class TestWriteToolLinting:
    @pytest.mark.asyncio
    async def test_write_warns_on_bad_syntax(self, tmp_path):
        """WriteTool should warn (but not fail) on syntax issues."""
        tool = WriteTool(workspace_dir=str(tmp_path))
        result = await tool.execute(
            path=str(tmp_path / "bad.py"),
            content="def foo(",  # Invalid
        )

        assert result.success  # WriteTool doesn't block
        assert "WARNING" in result.content
        assert "syntax" in result.content.lower()

    @pytest.mark.asyncio
    async def test_write_no_warning_on_valid(self, tmp_path):
        """WriteTool should not warn on valid files."""
        tool = WriteTool(workspace_dir=str(tmp_path))
        result = await tool.execute(
            path=str(tmp_path / "good.py"),
            content="x = 1\n",
        )

        assert result.success
        assert "WARNING" not in result.content
