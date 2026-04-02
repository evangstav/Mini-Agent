"""First-class git tools with structured output for better LLM understanding."""

import asyncio
import os
from typing import Any

from .base import Tool, ToolResult

# Max bytes for git output (100 KB)
_OUTPUT_LIMIT = 100_000


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... output truncated at {limit} bytes]"


async def _run_git(
    args: list[str], cwd: str | None = None, timeout: int = 30
) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        raise TimeoutError(f"git {args[0]} timed out after {timeout}s")
    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _detect_repo(workspace_dir: str | None) -> str | None:
    """Return the workspace dir if it contains a git repo, else None."""
    check_dir = workspace_dir or os.getcwd()
    # Walk up to find .git
    d = check_dir
    while True:
        if os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git")):
            return check_dir  # run commands from workspace, not .git parent
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


class GitStatusTool(Tool):
    """Show working tree status with structured output."""

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return (
            "Show git working tree status: current branch, ahead/behind count, "
            "staged/unstaged/untracked files. Use before committing to see what changed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self) -> ToolResult:
        cwd = _detect_repo(self.workspace_dir)
        if not cwd:
            return ToolResult(success=False, error="Not a git repository")
        try:
            # Get branch name
            rc, branch, _ = await _run_git(["branch", "--show-current"], cwd=cwd)
            branch = branch.strip() or "(detached HEAD)"

            # Get porcelain status for structured parsing
            rc, status_out, err = await _run_git(["status", "--porcelain=v2", "--branch"], cwd=cwd)
            if rc != 0:
                return ToolResult(success=False, error=err.strip())

            # Also get human-readable status for context
            rc2, readable, _ = await _run_git(["status", "--short"], cwd=cwd)

            lines = []
            lines.append(f"Branch: {branch}")

            # Parse ahead/behind from porcelain v2
            for line in status_out.splitlines():
                if line.startswith("# branch.ab"):
                    parts = line.split()
                    ahead = parts[2] if len(parts) > 2 else "+0"
                    behind = parts[3] if len(parts) > 3 else "-0"
                    lines.append(f"Ahead/Behind: {ahead} {behind}")
                    break

            if readable.strip():
                lines.append("")
                lines.append("Changes:")
                lines.append(readable.strip())
            else:
                lines.append("")
                lines.append("Working tree clean")

            return ToolResult(success=True, content="\n".join(lines))
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))


class GitDiffTool(Tool):
    """Show diffs with optional scope control."""

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "git_diff"

    @property
    def description(self) -> str:
        return (
            "Show git diff. Default: unstaged changes. "
            "Use staged=true for cached changes, ref='origin/main...HEAD' for branch diff, "
            "stat_only=true for file-level summary. Use path to limit scope."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Git ref or range (e.g. 'HEAD~3', 'origin/main...HEAD'). Default: working tree diff.",
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show staged (cached) changes instead of unstaged. Default: false.",
                    "default": False,
                },
                "path": {
                    "type": "string",
                    "description": "Limit diff to a specific file or directory path.",
                },
                "stat_only": {
                    "type": "boolean",
                    "description": "Show only file-level summary (--stat). Default: false.",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        ref: str | None = None,
        staged: bool = False,
        path: str | None = None,
        stat_only: bool = False,
    ) -> ToolResult:
        cwd = _detect_repo(self.workspace_dir)
        if not cwd:
            return ToolResult(success=False, error="Not a git repository")
        try:
            args = ["diff"]
            if staged:
                args.append("--cached")
            if stat_only:
                args.append("--stat")
            if ref:
                args.append(ref)
            if path:
                args.extend(["--", path])

            rc, out, err = await _run_git(args, cwd=cwd, timeout=30)
            if rc != 0:
                return ToolResult(success=False, error=err.strip())

            content = _truncate(out) if out.strip() else "(no changes)"
            return ToolResult(success=True, content=content)
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))


class GitCommitTool(Tool):
    """Create a git commit with safety checks."""

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "git_commit"

    @property
    def description(self) -> str:
        return (
            "Stage files and create a git commit. Provide specific files to stage, "
            "or add_all=true for everything. Fails if nothing is staged. "
            "Always run git_status first to see what will be committed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to stage before committing.",
                },
                "add_all": {
                    "type": "boolean",
                    "description": "Stage all modified and new files (git add -A). Default: false.",
                    "default": False,
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        message: str,
        files: list[str] | None = None,
        add_all: bool = False,
    ) -> ToolResult:
        cwd = _detect_repo(self.workspace_dir)
        if not cwd:
            return ToolResult(success=False, error="Not a git repository")
        try:
            # Stage files
            if add_all:
                rc, _, err = await _run_git(["add", "-A"], cwd=cwd)
                if rc != 0:
                    return ToolResult(success=False, error=f"git add -A failed: {err.strip()}")
            elif files:
                rc, _, err = await _run_git(["add", "--"] + files, cwd=cwd)
                if rc != 0:
                    return ToolResult(success=False, error=f"git add failed: {err.strip()}")

            # Check there's something to commit
            rc, staged, _ = await _run_git(["diff", "--cached", "--stat"], cwd=cwd)
            if not staged.strip():
                return ToolResult(success=False, error="Nothing staged to commit. Stage files first.")

            # Commit
            rc, out, err = await _run_git(["commit", "-m", message], cwd=cwd)
            if rc != 0:
                return ToolResult(success=False, error=err.strip())

            return ToolResult(success=True, content=out.strip())
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))


class GitLogTool(Tool):
    """Show commit history with structured output."""

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "git_log"

    @property
    def description(self) -> str:
        return (
            "Show commit history (default: 10 most recent). "
            "Use oneline=true for compact output, ref for ranges, path to filter by file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits to show. Default: 10.",
                    "default": 10,
                },
                "ref": {
                    "type": "string",
                    "description": "Git ref or range (e.g. 'origin/main..HEAD'). Default: current branch.",
                },
                "oneline": {
                    "type": "boolean",
                    "description": "Use compact one-line format. Default: false.",
                    "default": False,
                },
                "path": {
                    "type": "string",
                    "description": "Show only commits affecting this file/directory.",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        max_count: int = 10,
        ref: str | None = None,
        oneline: bool = False,
        path: str | None = None,
    ) -> ToolResult:
        cwd = _detect_repo(self.workspace_dir)
        if not cwd:
            return ToolResult(success=False, error="Not a git repository")
        try:
            args = ["log", f"--max-count={max_count}"]
            if oneline:
                args.append("--oneline")
            else:
                args.append("--format=%H%n%an <%ae>%n%ai%n%s%n")
            if ref:
                args.append(ref)
            if path:
                args.extend(["--", path])

            rc, out, err = await _run_git(args, cwd=cwd, timeout=30)
            if rc != 0:
                return ToolResult(success=False, error=err.strip())

            content = _truncate(out) if out.strip() else "(no commits)"
            return ToolResult(success=True, content=content)
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))


class GitBranchTool(Tool):
    """List, create, or switch branches."""

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "git_branch"

    @property
    def description(self) -> str:
        return (
            "List, create, or switch git branches. "
            "Default action: list. Use action='create' with branch_name to create, "
            "'switch' to checkout an existing branch."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "switch"],
                    "description": "Action to perform. Default: list.",
                    "default": "list",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (required for create/switch).",
                },
                "start_point": {
                    "type": "string",
                    "description": "Starting point for new branch (e.g. 'origin/main'). Only for create.",
                },
                "show_remote": {
                    "type": "boolean",
                    "description": "Include remote branches in list. Default: false.",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        action: str = "list",
        branch_name: str | None = None,
        start_point: str | None = None,
        show_remote: bool = False,
    ) -> ToolResult:
        cwd = _detect_repo(self.workspace_dir)
        if not cwd:
            return ToolResult(success=False, error="Not a git repository")
        try:
            if action == "list":
                args = ["branch", "-v"]
                if show_remote:
                    args.append("-a")
                rc, out, err = await _run_git(args, cwd=cwd)
                if rc != 0:
                    return ToolResult(success=False, error=err.strip())
                return ToolResult(success=True, content=out.strip() or "(no branches)")

            elif action == "create":
                if not branch_name:
                    return ToolResult(success=False, error="branch_name required for create action")
                args = ["checkout", "-b", branch_name]
                if start_point:
                    args.append(start_point)
                rc, out, err = await _run_git(args, cwd=cwd)
                if rc != 0:
                    return ToolResult(success=False, error=err.strip())
                return ToolResult(success=True, content=f"Created and switched to branch '{branch_name}'")

            elif action == "switch":
                if not branch_name:
                    return ToolResult(success=False, error="branch_name required for switch action")
                rc, out, err = await _run_git(["checkout", branch_name], cwd=cwd)
                if rc != 0:
                    return ToolResult(success=False, error=err.strip())
                return ToolResult(success=True, content=f"Switched to branch '{branch_name}'")

            else:
                return ToolResult(success=False, error=f"Unknown action: {action}. Use list/create/switch.")
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))
