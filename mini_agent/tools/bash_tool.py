"""Shell command execution tool with background process support."""

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
    re.compile(r"\bdd\b.*of=", re.IGNORECASE),
    re.compile(r"\bkillall\b"),
    re.compile(r"\bfdisk\b"),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bcurl\b.*\|\s*sh"),
    re.compile(r"\bwget\b.*\|\s*sh"),
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


class _BackgroundProcess:
    """Tracks a background subprocess and its accumulated output."""

    def __init__(self, process: asyncio.subprocess.Process, command: str):
        self.process = process
        self.command = command
        self._stdout_chunks: list[bytes] = []
        self._stderr_chunks: list[bytes] = []
        self._read_pos_out = 0
        self._read_pos_err = 0
        self._reader_task: asyncio.Task | None = None

    async def start_readers(self) -> None:
        """Spawn tasks that drain stdout/stderr into buffers."""
        self._reader_task = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        """Read from stdout and stderr concurrently until EOF."""
        async def _read_stream(
            stream: asyncio.StreamReader | None, buf: list[bytes]
        ) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    break
                buf.append(chunk)

        await asyncio.gather(
            _read_stream(self.process.stdout, self._stdout_chunks),
            _read_stream(self.process.stderr, self._stderr_chunks),
        )

    def get_new_output(self) -> str:
        """Return output accumulated since last call."""
        stdout = b"".join(self._stdout_chunks)
        stderr = b"".join(self._stderr_chunks)
        new_out = stdout[self._read_pos_out:]
        new_err = stderr[self._read_pos_err:]
        self._read_pos_out = len(stdout)
        self._read_pos_err = len(stderr)
        text = new_out.decode("utf-8", errors="replace")
        if new_err:
            text += f"\n[stderr]:\n{new_err.decode('utf-8', errors='replace')}"
        return _truncate_output(text) if text else "(no new output)"

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None


# Shared registry of background processes, keyed by integer ID.
_background_processes: dict[int, _BackgroundProcess] = {}
_next_pid = 1


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
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "Run the command in the background. Returns immediately "
                        "with a process_id. Use bash_output to read output, "
                        "bash_kill to stop the process."
                    ),
                    "default": False,
                },
            },
            "required": ["command"],
        }

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute. Returns (safe, reason)."""
        if self.allow_dangerous:
            return True, ""

        msg = _check_blocked(command)
        if msg:
            return False, msg
        return True, ""

    async def execute(
        self, command: str, timeout: int = 120, run_in_background: bool = False
    ) -> ToolResult:
        """Execute a shell command with safety checks."""
        global _next_pid

        # Security check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return ToolResult(
                success=False,
                error=f"Command blocked for safety: {reason}. Set allow_dangerous=True to bypass."
            )

        timeout = max(1, min(timeout, 600))

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

            if run_in_background:
                pid = _next_pid
                _next_pid += 1
                bg = _BackgroundProcess(process, command)
                _background_processes[pid] = bg
                await bg.start_readers()
                return ToolResult(
                    success=True,
                    content=f"Background process started (process_id={pid}). "
                    f"Use bash_output({pid}) to read output, bash_kill({pid}) to stop.",
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


class BashOutputTool(Tool):
    """Read recent output from a background bash process."""

    @property
    def name(self) -> str:
        return "bash_output"

    @property
    def description(self) -> str:
        return "Read new output from a background bash process."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "integer",
                    "description": "The process ID returned by bash with run_in_background.",
                },
            },
            "required": ["process_id"],
        }

    async def execute(self, process_id: int) -> ToolResult:
        bg = _background_processes.get(process_id)
        if bg is None:
            return ToolResult(
                success=False,
                error=f"No background process with id {process_id}. "
                f"Active: {list(_background_processes.keys()) or 'none'}",
            )
        output = bg.get_new_output()
        status = "running" if bg.is_running else f"exited (code {bg.process.returncode})"
        return ToolResult(success=True, content=f"[{status}]\n{output}")


class BashKillTool(Tool):
    """Stop a background bash process."""

    @property
    def name(self) -> str:
        return "bash_kill"

    @property
    def description(self) -> str:
        return "Stop a background bash process and return its remaining output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "integer",
                    "description": "The process ID returned by bash with run_in_background.",
                },
            },
            "required": ["process_id"],
        }

    async def execute(self, process_id: int) -> ToolResult:
        bg = _background_processes.get(process_id)
        if bg is None:
            return ToolResult(
                success=False,
                error=f"No background process with id {process_id}. "
                f"Active: {list(_background_processes.keys()) or 'none'}",
            )
        if bg.is_running:
            bg.process.terminate()
            try:
                await asyncio.wait_for(bg.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                bg.process.kill()
                await bg.process.wait()
        # Drain remaining output
        if bg._reader_task and not bg._reader_task.done():
            try:
                await asyncio.wait_for(bg._reader_task, timeout=2)
            except asyncio.TimeoutError:
                bg._reader_task.cancel()
        output = bg.get_new_output()
        del _background_processes[process_id]
        return ToolResult(
            success=True,
            content=f"Process {process_id} stopped (code {bg.process.returncode}).\n{output}",
        )
