"""Shell command execution tool."""

import asyncio
import platform
import re
from typing import Any

from .base import Tool, ToolResult


# Dangerous commands that could harm the system
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",  # Recursive delete from root
    r"\bdd\b.*of=",     # Direct disk write
    r"\bmkfs\b",        # Format filesystem
    r"\bfdisk\b",       # Partition tool
    r"\b shred\b",      # Secure delete
    r">\s*/dev/sd",     # Writing to disk device
    r"\bcurl\b.*\|\s*sh",  # Pipe to shell (common infection vector)
    r"\bwget\b.*\|\s*sh",  # Same for wget
]


class BashTool(Tool):
    """Execute shell commands with basic safety filtering."""

    def __init__(self, workspace_dir: str | None = None, allow_dangerous: bool = False):
        self.is_windows = platform.system() == "Windows"
        self.workspace_dir = workspace_dir
        self.allow_dangerous = allow_dangerous

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout/stderr."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120, max: 600).",
                    "default": 120,
                },
            },
            "required": ["command"],
        }

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute. Returns (safe, reason)."""
        if self.allow_dangerous:
            return True, ""

        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern: {pattern}"
        return True, ""

    async def execute(self, command: str, timeout: int = 120) -> ToolResult:
        """Execute a shell command with safety checks."""
        # Security check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return ToolResult(
                success=False,
                error=f"Command blocked for safety: {reason}. Set allow_dangerous=True to bypass."
            )

        timeout = max(1, min(timeout, 600))

        try:
            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe", "-NoProfile", "-Command", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(success=False, error=f"Command timed out after {timeout}s")

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            output = stdout_text
            if stderr_text:
                output += f"\n[stderr]:\n{stderr_text}"

            is_success = process.returncode == 0
            return ToolResult(
                success=is_success,
                content=output or "(no output)",
                error=None if is_success else f"Exit code {process.returncode}",
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))
