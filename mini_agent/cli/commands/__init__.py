"""Slash command framework."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from rich.console import Console

if TYPE_CHECKING:
    from ...agent import Agent
    from ...schema import LLMProvider
    from ..render.events import EventRenderer


@dataclass
class REPLContext:
    """Shared mutable state passed to slash commands."""

    agent: Agent
    console: Console
    renderer: EventRenderer
    model: str
    api_key: str
    provider_enum: Any  # LLMProvider
    api_base: str
    forks: dict[int, Agent]
    current_fork_id: int
    next_fork_id: int
    cancel_event: asyncio.Event
    workspace: str
    agent_tool: Any = None
    dream: Any = None


class SlashCommand(Protocol):
    """Protocol for slash commands."""

    @property
    def name(self) -> str: ...

    @property
    def aliases(self) -> list[str]: ...

    @property
    def description(self) -> str: ...

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        """Execute command. Return True to continue REPL, False to exit."""
        ...


class SlashCommandRegistry:
    """Registry for slash commands with name and alias lookup."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._by_name: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._by_name[cmd.name] = cmd
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def all_descriptions(self) -> dict[str, str]:
        """Return {name: description} for all commands (no aliases)."""
        return {cmd.name: cmd.description for cmd in self._by_name.values()}

    def all_names(self) -> list[str]:
        """All command names including aliases."""
        return list(self._commands.keys())
