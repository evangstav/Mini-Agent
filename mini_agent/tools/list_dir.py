"""List directory tool — show repo structure at a glance.

Research basis: Top agents provide directory listing as a first-class tool.
Our agent had to use bash with ls/find, wasting a tool call on something basic.
"""

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

_MAX_ENTRIES = 200


class ListDirTool(Tool):
    """List directory contents with tree-like structure."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "Show directory structure as a tree. Default depth is 2. "
            "Use this first to understand a project's layout before diving into files. "
            "Skips hidden dirs, __pycache__, node_modules, and .venv."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: workspace root).",
                },
                "depth": {
                    "type": "integer",
                    "description": "Max depth to recurse (default: 2, max: 5).",
                },
            },
            "required": [],
        }

    async def execute(self, path: str | None = None, depth: int = 2) -> ToolResult:
        try:
            target = Path(path) if path else self.workspace_dir
            if not target.is_absolute():
                target = self.workspace_dir / target

            try:
                target.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return ToolResult(success=False, error=f"Path escapes workspace: {target}")

            if not target.exists():
                return ToolResult(success=False, error=f"Directory not found: {target}")
            if not target.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {target}")

            depth = max(1, min(depth, 5))
            lines = []
            count = [0]
            _build_tree(target, "", depth, lines, count, self.workspace_dir)

            if not lines:
                return ToolResult(success=True, content="(empty directory)")

            rel = target.relative_to(self.workspace_dir) if target != self.workspace_dir else Path(".")
            header = f"{rel}/\n"
            result = header + "\n".join(lines)

            if count[0] >= _MAX_ENTRIES:
                result += f"\n\n(truncated at {_MAX_ENTRIES} entries — use a narrower path or lower depth)"

            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs", "*.egg-info",
})


def _should_skip(name: str) -> bool:
    return name.startswith(".") or name in _SKIP_DIRS or name.endswith(".egg-info")


def _build_tree(
    directory: Path, prefix: str, depth: int,
    lines: list[str], count: list[int], workspace: Path,
) -> None:
    if count[0] >= _MAX_ENTRIES:
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    # Filter
    entries = [e for e in entries if not _should_skip(e.name)]

    for i, entry in enumerate(entries):
        if count[0] >= _MAX_ENTRIES:
            return

        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{suffix}")
        count[0] += 1

        if entry.is_dir() and depth > 1:
            extension = "    " if is_last else "│   "
            _build_tree(entry, prefix + extension, depth - 1, lines, count, workspace)
