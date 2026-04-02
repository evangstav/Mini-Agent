"""Async REPL loop — the core interactive experience."""

import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from ..events import AgentCancelled, AgentDone, AgentError
from ..tools.mcp_loader import cleanup_mcp_connections
from .commands import REPLContext
from .completions import SlashCommandCompleter
from .render import console
from .session import save_session
from .setup import build_command_registry
from .status_bar import make_bottom_toolbar


async def launch_repl(ctx: REPLContext) -> None:
    """Launch the interactive REPL."""
    registry = build_command_registry()
    # Attach registry to context so HelpCommand can access it
    ctx._registry = registry  # type: ignore[attr-defined]

    history_path = Path(ctx.workspace) / ".runtime" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    kb = KeyBindings()

    @kb.add(Keys.Escape, Keys.Enter)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=SlashCommandCompleter(registry),
        bottom_toolbar=make_bottom_toolbar(ctx),
        key_bindings=kb,
    )

    # Banner
    console.print()
    console.print(f"[heading]Mini-Agent[/] [info]v0.1.0[/]")
    console.print(f"[info]Model:[/] {ctx.model}  [info]Provider:[/] {ctx.provider_enum.value}")
    console.print(f"[info]Workspace:[/] {ctx.workspace}")
    console.print(f"[info]Type /help for commands. Ctrl+C to cancel. Ctrl+D to exit.[/]")
    console.print()

    while True:
        prompt_text = FormattedText([
            ("class:gray", ""),
            ("class:prompt", "> "),
        ])

        try:
            user_input = await session.prompt_async(prompt_text, multiline=False)
        except KeyboardInterrupt:
            continue
        except EOFError:
            auto_save_path = str(Path(ctx.workspace) / ".runtime" / "autosave_session.json")
            if len(ctx.agent.messages) > 1:
                save_session(ctx.agent.messages, auto_save_path)
                console.print(f"\n[info]Session auto-saved to[/] {auto_save_path}")
            console.print("[info]Goodbye![/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd_name = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            cmd = registry.get(cmd_name)
            if cmd:
                should_continue = await cmd.execute(arg, ctx)
                if not should_continue:
                    break
            else:
                console.print(f"[error]Unknown command:[/] {cmd_name}. Type /help for commands.")
            continue

        # Send to agent
        ctx.agent.add_user_message(user_input)
        ctx.cancel_event.clear()
        start_time = time.monotonic()

        try:
            async for event in ctx.agent.run_stream(cancel_event=ctx.cancel_event):
                ctx.renderer.render(event)

                if isinstance(event, AgentDone):
                    elapsed = time.monotonic() - start_time
                    ctx.renderer.render_done(event, elapsed, ctx.model, ctx.agent.token_usage)

                elif isinstance(event, AgentCancelled):
                    elapsed = time.monotonic() - start_time
                    ctx.renderer.render_cancelled(event, elapsed)

                elif isinstance(event, AgentError):
                    pass  # already rendered by renderer.render()

        except KeyboardInterrupt:
            ctx.cancel_event.set()
            console.print("\n[warning]Cancelled.[/]")

    await ctx.agent.end_session()
    await cleanup_mcp_connections()
