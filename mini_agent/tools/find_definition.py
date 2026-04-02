"""Find definition tool — locate classes, functions, and methods by name using Python AST.

Research basis: SWE-Agent gap analysis showed agents waste 3-5 steps on regex
guesswork to find definitions. A dedicated AST-based tool saves those steps.
"""

import ast
import os
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

_MAX_RESULTS = 20


class FindDefinitionTool(Tool):
    """Find Python class, function, or method definitions by name."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "find_definition"

    @property
    def description(self) -> str:
        return (
            "Find Python class, function, or method definitions by name using AST parsing. "
            "Returns file path, line number, and signature for each match. "
            "Much faster and more reliable than grep for finding where something is defined. "
            "Use this instead of grep when looking for a specific class or function."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to search for (class, function, or method name). Partial matches supported.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: workspace root). Searches recursively.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["all", "class", "function", "method"],
                    "description": "Filter by definition type (default: all).",
                },
            },
            "required": ["name"],
        }

    async def execute(
        self, name: str, path: str | None = None, kind: str = "all"
    ) -> ToolResult:
        try:
            search_dir = Path(path) if path else self.workspace_dir
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            try:
                search_dir.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return ToolResult(success=False, error=f"Path escapes workspace: {search_dir}")

            if not search_dir.exists():
                return ToolResult(success=False, error=f"Directory not found: {search_dir}")

            results = []
            for py_file in sorted(search_dir.rglob("*.py")):
                # Skip hidden dirs, __pycache__, .venv, etc.
                parts = py_file.relative_to(self.workspace_dir).parts
                if any(p.startswith(".") or p == "__pycache__" or p == "node_modules" for p in parts):
                    continue

                try:
                    source = py_file.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(source, filename=str(py_file))
                except (SyntaxError, UnicodeDecodeError):
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and kind in ("all", "class"):
                        if name.lower() in node.name.lower():
                            rel_path = py_file.relative_to(self.workspace_dir)
                            bases = ", ".join(
                                ast.unparse(b) for b in node.bases
                            ) if node.bases else ""
                            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
                            results.append(f"{rel_path}:{node.lineno}  {sig}")

                            # Also find methods inside this class
                            if kind in ("all", "method"):
                                for item in node.body:
                                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                        if name.lower() in item.name.lower() or name.lower() in node.name.lower():
                                            args = _format_args(item.args)
                                            prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                                            results.append(
                                                f"{rel_path}:{item.lineno}    {prefix} {node.name}.{item.name}({args})"
                                            )

                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and kind in ("all", "function"):
                        # Only top-level functions (not methods inside classes)
                        if name.lower() in node.name.lower():
                            # Check it's not inside a class
                            rel_path = py_file.relative_to(self.workspace_dir)
                            args = _format_args(node.args)
                            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                            results.append(f"{rel_path}:{node.lineno}  {prefix} {node.name}({args})")

                if len(results) >= _MAX_RESULTS:
                    break

            if not results:
                return ToolResult(success=True, content=f"No definitions matching '{name}' found.")

            output = "\n".join(results[:_MAX_RESULTS])
            if len(results) > _MAX_RESULTS:
                output += f"\n\n(showing {_MAX_RESULTS} of {len(results)} matches)"
            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _format_args(args: ast.arguments) -> str:
    """Format function arguments concisely."""
    parts = []
    for arg in args.args:
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{annotation}")
    if len(parts) > 5:
        return ", ".join(parts[:3]) + f", ... ({len(parts)} args)"
    return ", ".join(parts)
