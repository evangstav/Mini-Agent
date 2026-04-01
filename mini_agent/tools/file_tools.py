"""File operation tools."""

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


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


class ReadTool(Tool):
    """Read file content."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file contents with line numbers. Supports offset/limit for large files."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "Starting line number (1-indexed)"},
                "limit": {"type": "integer", "description": "Number of lines to read"},
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

            start = (offset - 1) if offset else 0
            end = (start + limit) if limit else len(lines)
            start = max(0, start)
            end = min(end, len(lines))

            numbered = [f"{i:6d}|{line.rstrip(chr(10))}" for i, line in enumerate(lines[start:end], start=start + 1)]
            return ToolResult(success=True, content="\n".join(numbered))
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
        return "Write content to a file (overwrites existing). Read first before editing."

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
            return ToolResult(success=True, content=f"Successfully wrote to {file_path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EditTool(Tool):
    """Edit file by replacing text."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace exact string in a file. old_str must be unique in the file."

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

            file_path.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
            return ToolResult(success=True, content=f"Successfully edited {file_path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
