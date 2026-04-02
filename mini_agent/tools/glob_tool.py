"""Glob tool — fast file pattern matching."""

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

MAX_RESULTS = 500


class GlobTool(Tool):
    """Find files by glob pattern."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "Find files by glob pattern. Returns paths sorted by modification time (newest first). "
            "Use '**/*.py' for recursive Python files, 'src/*.ts' for flat matches. "
            "Max 500 results. Use this instead of 'bash find' for file discovery."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files (e.g. '**/*.py')"},
                "path": {"type": "string", "description": "Base directory to search in (default: workspace root)"},
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str | None = None) -> ToolResult:
        try:
            base = Path(path) if path else self.workspace_dir
            if not base.is_absolute():
                base = self.workspace_dir / base

            # Ensure base is inside workspace
            try:
                base.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return ToolResult(success=False, error=f"Path escapes workspace: {base}")

            if not base.exists():
                return ToolResult(success=False, error=f"Directory not found: {base}")

            matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            # Filter to files only
            matches = [m for m in matches if m.is_file()]

            if not matches:
                return ToolResult(success=True, content="No matches found.")

            truncated = len(matches) > MAX_RESULTS
            matches = matches[:MAX_RESULTS]

            # Show paths relative to workspace
            lines = [str(m.relative_to(self.workspace_dir)) for m in matches]
            result = "\n".join(lines)
            if truncated:
                result += f"\n\n(truncated — showing {MAX_RESULTS} of {len(matches)}+ matches)"

            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
