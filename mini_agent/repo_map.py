"""Repo map — generate a compact codebase skeleton for context injection.

Parses Python files with ast to extract class/function signatures and builds
a budget-constrained skeleton that gives the LLM structural awareness of the
codebase without consuming the full context window.

Research basis: Aider's tree-sitter repo map achieves 10x token efficiency
for navigation (4-6% context window vs 50-70% for iterative search).
This is the Python-only v1; tree-sitter upgrade planned (hq-lsnt).
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories to skip
_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs", ".runtime", ".claude",
})

# File patterns to skip
_SKIP_PATTERNS = frozenset({
    "setup.py", "conftest.py", "__init__.py",
})


def _should_include(path: Path, project_dir: Path) -> bool:
    """Check if a Python file should be included in the repo map."""
    rel = path.relative_to(project_dir)
    parts = rel.parts

    # Skip hidden dirs and known noise dirs
    if any(p.startswith(".") or p in _SKIP_DIRS for p in parts):
        return False

    # Skip test files (they're not helpful for understanding structure)
    if any(p.startswith("test") for p in parts):
        return False
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return False

    # Skip trivial files
    if path.name in _SKIP_PATTERNS:
        return False

    return True


def _sort_key(path: Path, project_dir: Path) -> tuple:
    """Sort files by relevance: shallow > deep, src/ > other."""
    rel = path.relative_to(project_dir)
    depth = len(rel.parts)
    # Boost src/ and lib/ directories
    is_src = rel.parts[0] in ("src", "lib") if rel.parts else False
    return (not is_src, depth, str(rel))


def _format_args(args: ast.arguments, max_args: int = 4) -> str:
    """Format function arguments concisely."""
    parts = []
    for arg in args.args:
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{annotation}")
    if len(parts) > max_args:
        return ", ".join(parts[:max_args - 1]) + ", ..."
    return ", ".join(parts)


def _format_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format return type annotation if present."""
    if node.returns:
        return f" -> {ast.unparse(node.returns)}"
    return ""


def _extract_file_skeleton(source: str, filepath: str) -> list[str]:
    """Extract class and function signatures from a Python file."""
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    lines = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            lines.append(f"  {sig}")

            # Extract methods (not private ones unless they're __init__)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    args = _format_args(item.args)
                    ret = _format_return(item)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    lines.append(f"    {prefix}def {item.name}({args}){ret}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions (skip private)
            if node.name.startswith("_"):
                continue
            args = _format_args(node.args)
            ret = _format_return(node)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            lines.append(f"  {prefix}def {node.name}({args}){ret}")

    return lines


def generate_repo_map(
    project_dir: str,
    max_chars: int = 6000,
    cache: bool = True,
) -> str:
    """Generate a compact codebase skeleton for context injection.

    Args:
        project_dir: Root directory of the project.
        max_chars: Maximum characters for the skeleton output.
        cache: If True, cache to .runtime/repo_map.md and reuse if fresh.

    Returns:
        Markdown-formatted skeleton string.
    """
    root = Path(project_dir).resolve()

    # Check cache
    if cache:
        cache_path = root / ".runtime" / "repo_map.md"
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            py_files = [f for f in root.rglob("*.py") if _should_include(f, root)]
            newest = max((f.stat().st_mtime for f in py_files), default=0) if py_files else 0
            if newest <= cache_mtime:
                logger.debug("Repo map cache hit: %s", cache_path)
                content = cache_path.read_text(encoding="utf-8")
                if len(content) <= max_chars:
                    return content

    # Generate fresh
    skeleton = _build_skeleton(root, max_chars)

    # Write cache
    if cache and skeleton:
        cache_path = root / ".runtime" / "repo_map.md"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(skeleton, encoding="utf-8")
        logger.debug("Repo map cached to: %s", cache_path)

    return skeleton


def _build_skeleton(root: Path, max_chars: int) -> str:
    """Build the skeleton string within the character budget."""
    py_files = [f for f in root.rglob("*.py") if _should_include(f, root)]
    py_files.sort(key=lambda f: _sort_key(f, root))

    sections = []
    total_chars = 0

    for filepath in py_files:
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lines = _extract_file_skeleton(source, str(filepath))
        if not lines:
            continue

        rel_path = filepath.relative_to(root)
        section = f"## {rel_path}\n" + "\n".join(lines)

        if total_chars + len(section) + 2 > max_chars:
            # Budget exhausted — add truncation notice
            remaining = len(py_files) - len(sections)
            if remaining > 0:
                sections.append(f"\n(... {remaining} more files not shown)")
            break

        sections.append(section)
        total_chars += len(section) + 2  # +2 for separating newlines

    return "\n\n".join(sections) if sections else ""
