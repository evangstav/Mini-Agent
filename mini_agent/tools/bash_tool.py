"""Shell command execution tool."""

import asyncio
import os
import platform
import re
from typing import Any

from .base import Tool, ToolResult

# Env var names that likely contain secrets — stripped from subprocess env
_SECRET_PATTERNS = re.compile(
    r"(API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY|AUTH)",
    re.IGNORECASE,
)

# Max bytes for stdout/stderr (100 KB)
_OUTPUT_LIMIT = 100_000

# Commands that should never be executed — destructive or system-level only.
# chmod and sudo are intentionally allowed for common dev operations.
_BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f", re.IGNORECASE),  # rm -rf variants
    re.compile(r"\brm\s+-[^\s]*f[^\s]*r", re.IGNORECASE),  # rm -fr variants
    re.compile(r"\bshred\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r"\bkillall\b"),
]


def _sanitize_env() -> dict[str, str]:
    """Return a copy of os.environ with secret-looking vars removed."""
    return {k: v for k, v in os.environ.items() if not _SECRET_PATTERNS.search(k)}


def _check_blocked(command: str) -> str | None:
    """Return an error message if the command matches a blocked pattern."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"Command blocked by security policy: matches pattern {pattern.pattern!r}"
    return None


def _truncate_output(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    """Truncate output to limit bytes with notice."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... output truncated at {limit} bytes]"


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

        # Check blocklist
        if blocked_msg := _check_blocked(command):
            return ToolResult(success=False, error=blocked_msg)

        safe_env = _sanitize_env()

        try:
            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe", "-NoProfile", "-Command", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                    env=safe_env,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                    env=safe_env,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(success=False, error=f"Command timed out after {timeout}s")

            stdout_text = _truncate_output(stdout.decode("utf-8", errors="replace"))
            stderr_text = _truncate_output(stderr.decode("utf-8", errors="replace"))
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
