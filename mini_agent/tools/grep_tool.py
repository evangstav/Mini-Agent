"""Grep tool — content search with ripgrep fallback to grep."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

MAX_OUTPUT_BYTES = 100_000  # 100KB cap


class GrepTool(Tool):
    """Search file contents by regex pattern."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents by regex pattern. Uses ripgrep if available, falls back to grep. "
            "Returns matching lines with file paths and line numbers. "
            "Use this instead of 'bash grep' — it respects workspace boundaries and caps output at 100KB. "
            "Supports glob filters (e.g., glob='*.py'), case-insensitive search, and context lines."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search (default: workspace root)"},
                "glob": {"type": "string", "description": "Glob filter for files (e.g. '*.py')"},
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode (default: content)",
                },
                "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                "context": {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "content",
        case_insensitive: bool = False,
        context: int | None = None,
    ) -> ToolResult:
        try:
            search_path = Path(path) if path else self.workspace_dir
            if not search_path.is_absolute():
                search_path = self.workspace_dir / search_path

            # Ensure path is inside workspace
            try:
                search_path.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return ToolResult(success=False, error=f"Path escapes workspace: {search_path}")

            if not search_path.exists():
                return ToolResult(success=False, error=f"Path not found: {search_path}")

            use_rg = shutil.which("rg") is not None
            cmd = self._build_cmd(
                use_rg, pattern, search_path, glob, output_mode, case_insensitive, context,
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir),
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode("utf-8", errors="replace")

            # Cap output size
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n\n(output truncated at 100KB)"

            if proc.returncode == 1 and not output:
                return ToolResult(success=True, content="No matches found.")

            if proc.returncode not in (0, 1):
                err = stderr.decode("utf-8", errors="replace")
                return ToolResult(success=False, error=f"Search failed (exit {proc.returncode}): {err}")

            return ToolResult(success=True, content=output.rstrip())
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _build_cmd(
        use_rg: bool,
        pattern: str,
        search_path: Path,
        glob: str | None,
        output_mode: str,
        case_insensitive: bool,
        context: int | None,
    ) -> list[str]:
        if use_rg:
            cmd = ["rg", "--no-heading", "--line-number"]
            if case_insensitive:
                cmd.append("-i")
            if glob:
                cmd.extend(["--glob", glob])
            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            if context is not None and output_mode == "content":
                cmd.extend(["-C", str(context)])
            cmd.extend([pattern, str(search_path)])
        else:
            cmd = ["grep", "-rn"]
            if case_insensitive:
                cmd.append("-i")
            if glob:
                cmd.extend(["--include", glob])
            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            if context is not None and output_mode == "content":
                cmd.extend(["-C", str(context)])
            cmd.extend([pattern, str(search_path)])

        return cmd
