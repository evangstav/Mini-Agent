"""Shared Rich Console instance and theme."""

from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "tool.name": "bold cyan",
    "tool.args": "dim",
    "tool.success": "green",
    "tool.error": "bold red",
    "tool.preview": "dim",
    "thinking": "dim italic",
    "prompt.token": "dim",
    "info": "dim",
    "info.label": "dim",
    "info.value": "default",
    "warning": "bold yellow",
    "error": "bold red",
    "heading": "bold cyan",
    "separator": "dim",
    "success": "green",
    "command": "green",
    "command.desc": "default",
})

console = Console(theme=THEME, highlight=False)
