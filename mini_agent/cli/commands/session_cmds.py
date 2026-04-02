"""Session commands: /save, /load, /clear, /compact."""

from pathlib import Path

from . import REPLContext
from ..session import load_session, save_session


class SaveCommand:
    name = "/save"
    aliases: list[str] = []
    description = "Save session to file (/save [path])"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        path = arg or "session.json"
        save_session(ctx.agent.messages, path)
        ctx.console.print(f"[success]Session saved to[/] {path}")
        return True


class LoadCommand:
    name = "/load"
    aliases: list[str] = []
    description = "Load session from file (/load <path>)"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        if not arg:
            ctx.console.print("[error]Usage: /load <path>[/]")
            return True
        if not Path(arg).exists():
            ctx.console.print(f"[error]File not found:[/] {arg}")
            return True
        try:
            ctx.agent.messages = load_session(arg)
        except ValueError as e:
            ctx.console.print(f"[error]Error loading session:[/] {e}")
            return True
        ctx.console.print(f"[success]Session loaded from[/] {arg}")
        return True


class ClearCommand:
    name = "/clear"
    aliases: list[str] = []
    description = "Clear conversation history (keep system prompt)"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        system_msg = ctx.agent.messages[0] if ctx.agent.messages and ctx.agent.messages[0].role == "system" else None
        ctx.agent.messages = [system_msg] if system_msg else []
        ctx.console.print("[success]Conversation cleared.[/]")
        return True


class CompactCommand:
    name = "/compact"
    aliases: list[str] = []
    description = "Force context compaction"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        from ...context import compact_messages, estimate_tokens

        before = estimate_tokens(ctx.agent.messages)
        ctx.agent.messages = await compact_messages(
            ctx.agent.messages, ctx.agent.llm, token_threshold=0
        )
        after = estimate_tokens(ctx.agent.messages)
        from ..render.tables import _format_tokens
        ctx.console.print(f"[success]Compacted:[/] {_format_tokens(before)} -> {_format_tokens(after)} tokens")
        return True
