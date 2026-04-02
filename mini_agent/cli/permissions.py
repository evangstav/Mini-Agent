"""Permission prompting with Rich output."""

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


class PermissionManager:
    """Manages tool permission prompts (y/n/always)."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._always_allowed: set[str] = set()

    async def check(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name in self._always_allowed:
            return True

        args_preview = json.dumps(arguments, indent=2)
        if len(args_preview) > 300:
            args_preview = args_preview[:300] + "..."

        self._console.print(
            Panel(
                Syntax(args_preview, "json", theme="monokai", line_numbers=False),
                title=f"[bold yellow]Permission: [bold cyan]{tool_name}[/][/]",
                border_style="yellow",
            )
        )

        while True:
            try:
                answer = input("[y]es / [n]o / [a]lways: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                self._always_allowed.add(tool_name)
                return True
