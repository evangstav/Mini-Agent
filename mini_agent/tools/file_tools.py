"""File operation tools."""

import ast
import difflib
import json
import subprocess
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

_MAX_DIFF_LINES = 60


def _unified_diff(old: str, new: str, filepath: str) -> str:
    """Generate a compact unified diff, capped at _MAX_DIFF_LINES."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=filepath, tofile=filepath, n=3))
    if not diff:
        return "(no changes)"
    if len(diff) > _MAX_DIFF_LINES:
        remaining = len(diff) - _MAX_DIFF_LINES
        diff = diff[:_MAX_DIFF_LINES]
        diff.append(f"\n[... {remaining} more diff lines ...]\n")
    return "".join(diff).rstrip()


# Linter-integrated editing: validate syntax after modifications.
# SWE-Agent research (NeurIPS 2024) shows rejecting invalid edits prevents cascading failures.

_LINTERS: dict[str, Any] = {}  # extension -> checker function


def _check_python_syntax(path: Path, content: str) -> str | None:
    """Check Python syntax. Returns error message or None if valid."""
    try:
        ast.parse(content, filename=str(path))
        return None
    except SyntaxError as e:
        return f"Python syntax error at line {e.lineno}: {e.msg}"


def _check_json_syntax(path: Path, content: str) -> str | None:
    """Check JSON syntax. Returns error message or None if valid."""
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return f"JSON syntax error at line {e.lineno}: {e.msg}"


def _lint_file(path: Path, content: str) -> str | None:
    """Run syntax check based on file extension. Returns error or None."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _check_python_syntax(path, content)
    elif suffix == ".json":
        return _check_json_syntax(path, content)
    return None  # No linter for this file type


def _validate_path(file_path: Path, workspace_dir: Path) -> str | None:
    """Validate that resolved path is inside workspace_dir.

    Returns an error string if the path escapes the workspace, None if safe.
    """
    try:
        resolved = file_path.resolve()
        workspace_resolved = workspace_dir.resolve()
        # Ensure the resolved path starts with the workspace directory
        resolved.relative_to(workspace_resolved)
        return None
    except ValueError:
        return f"Path escapes workspace: {file_path} resolves outside {workspace_dir}"


# Default line cap for read_file output (SWE-Agent research: windowed output improves agent performance)
_DEFAULT_MAX_LINES = 200


class ReadTool(Tool):
    """Read file content."""

    def __init__(self, workspace_dir: str = ".", max_lines: int = _DEFAULT_MAX_LINES):
        self.workspace_dir = Path(workspace_dir).absolute()
        self.max_lines = max_lines

    @property
    def read_only(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents with line numbers. Returns up to 200 lines by default. "
            "For large files, use offset and limit to read specific sections. "
            "If the file exceeds the limit, the output is truncated with a notice showing total line count."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or workspace-relative path to the file."},
                "offset": {"type": "integer", "description": "Starting line number (1-indexed). Use with limit to read a specific section."},
                "limit": {"type": "integer", "description": f"Max lines to return (default: {self.max_lines}). Set higher if you need more context."},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if error := _validate_path(file_path, self.workspace_dir):
                return ToolResult(success=False, error=error)

            if not file_path.exists():
                return ToolResult(success=False, error=f"File not found: {path}")

            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start = (offset - 1) if offset else 0
            effective_limit = limit if limit is not None else self.max_lines
            end = start + effective_limit
            start = max(0, start)
            end = min(end, total_lines)

            numbered = [f"{i:6d}|{line.rstrip(chr(10))}" for i, line in enumerate(lines[start:end], start=start + 1)]
            result = "\n".join(numbered)

            # Add truncation notice if output was capped
            if end < total_lines and limit is None:
                result += (
                    f"\n\n[Showing lines {start + 1}-{end} of {total_lines}. "
                    f"Use offset={end + 1} to see more, or limit=N to read a specific range.]"
                )

            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteTool(Tool):
    """Write content to a file."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file, creating parent directories if needed. "
            "OVERWRITES the entire file. Use edit_file for surgical changes to existing files. "
            "Always read_file first to verify you have the current content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if error := _validate_path(file_path, self.workspace_dir):
                return ToolResult(success=False, error=error)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            # Lint check: warn (but don't fail) for syntax issues in written files
            lint_error = _lint_file(file_path, content)
            warning = f" WARNING: {lint_error}" if lint_error else ""
            return ToolResult(success=True, content=f"Successfully wrote to {file_path}{warning}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# Shared edit history for undo support (file path → previous content)
_edit_history: dict[str, str] = {}


class EditTool(Tool):
    """Edit file by replacing text."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Replace an exact string in a file with a new string. "
            "old_str must appear EXACTLY ONCE in the file (including whitespace and indentation). "
            "If it matches 0 or 2+ times, the edit fails. Include enough surrounding context "
            "to make old_str unique. Read the file first to get the exact text."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "old_str": {"type": "string", "description": "Exact string to find"},
                "new_str": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def execute(self, path: str, old_str: str, new_str: str) -> ToolResult:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if error := _validate_path(file_path, self.workspace_dir):
                return ToolResult(success=False, error=error)

            if not file_path.exists():
                return ToolResult(success=False, error=f"File not found: {path}")

            content = file_path.read_text(encoding="utf-8")
            if old_str not in content:
                return ToolResult(success=False, error=f"Text not found in file: {old_str}")

            match_count = content.count(old_str)
            if match_count > 1:
                return ToolResult(
                    success=False,
                    error=f"old_str matches {match_count} times in {path} — must be unique. "
                          f"Provide more surrounding context to disambiguate.",
                )

            new_content = content.replace(old_str, new_str, 1)

            # Save for undo before writing
            _edit_history[str(file_path)] = content

            # Lint check: check for NEW syntax errors (SWE-Agent approach: warn but write)
            old_lint = _lint_file(file_path, content)
            new_lint = _lint_file(file_path, new_content)
            lint_warning = ""
            if new_lint and new_lint != old_lint:
                lint_warning = f" WARNING: {new_lint}"

            file_path.write_text(new_content, encoding="utf-8")

            # Show diff so the LLM sees exactly what changed
            diff = _unified_diff(content, new_content, str(file_path.relative_to(self.workspace_dir)))

            # Verification nudge for code files
            verify_hint = ""
            if file_path.suffix in (".py", ".js", ".ts", ".go", ".rs"):
                verify_hint = (
                    "\nSelf-check: does this change fix the ROOT CAUSE? "
                    "Run tests to verify."
                )

            return ToolResult(success=True, content=f"Edited {file_path}{lint_warning}\n\n{diff}{verify_hint}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class UndoEditTool(Tool):
    """Undo the last edit to a file."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "undo_edit"

    @property
    def description(self) -> str:
        return (
            "Revert a file to its state before the last edit_file call. "
            "Only one level of undo is available per file. "
            "Use when an edit was wrong and you want to start over."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to undo."},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            key = str(file_path)
            if key not in _edit_history:
                return ToolResult(success=False, error=f"No edit to undo for: {path}")

            previous = _edit_history.pop(key)
            file_path.write_text(previous, encoding="utf-8")
            return ToolResult(success=True, content=f"Reverted {file_path} to previous state.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
