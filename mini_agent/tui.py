"""TUI — Rich REPL interface for Mini-Agent.

Natural terminal prompt (prompt_toolkit) + rich-formatted output.
No full-screen takeover. Your terminal, your scrollback, your background.

Controls:
  Enter      → send message
  Ctrl+C     → cancel running agent
  Ctrl+D     → quit
  Esc+Enter  → newline (multi-line input)
  /help      → slash commands
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from .agent import Agent
from .context import ToolResultStore, compact_messages, estimate_tokens
from .dream import DreamConsolidator
from .events import (
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .hooks import HookRegistry
from .llm import LLMClient
from .schema import LLMProvider, Message
from .tools.bash_tool import BashTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.glob_tool import GlobTool
from .tools.grep_tool import GrepTool


# ── Rich console with custom theme ──────────────────────────────────────────

_THEME = Theme({
    "info":      "dim",
    "user":      "bold cyan",
    "assistant": "default",
    "tool":      "#a78bfa",
    "tool.ok":   "#a6e3a1",
    "tool.err":  "#f38ba8 bold",
    "thinking":  "dim italic",
    "done":      "dim",
    "err":       "#f38ba8 bold",
    "prompt":    "bold cyan",
})

console = Console(theme=_THEME, highlight=False)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_session(messages: list[Message], path: str) -> None:
    data = [msg.model_dump() for msg in messages]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _load_session(path: str) -> list[Message]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Message(**item) for item in raw]


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s


# ── Permission manager ──────────────────────────────────────────────────────

class PermissionManager:
    """Manages tool permission prompts (y/n/a)."""

    def __init__(self):
        self._always_allowed: set[str] = set()

    async def check(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name in self._always_allowed:
            return True

        args_preview = json.dumps(arguments, indent=2)
        if len(args_preview) > 300:
            args_preview = args_preview[:300] + "..."

        console.print()
        console.print(
            Panel(
                Text(args_preview, style="dim"),
                title=f"[bold yellow]allow[/] [bold cyan]{tool_name}[/] [bold yellow]?[/]",
                title_align="left",
                border_style="#e2b714",
                padding=(0, 1),
            )
        )

        while True:
            try:
                answer = input("  [y]es / [n]o / [a]lways: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                self._always_allowed.add(tool_name)
                return True


# ── Session persistence ──────────────────────────────────────────────────────

def save_session(messages: list[Message], path: str) -> None:
    _save_session(messages, path)


def load_session(path: str) -> list[Message]:
    return _load_session(path)


# ── Main REPL ────────────────────────────────────────────────────────────────

async def run_tui(
    api_key: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    api_base: str | None = None,
    workspace: str | None = None,
    system_prompt: str | None = None,
    max_steps: int = 50,
    session_file: str | None = None,
    enable_permissions: bool = True,
) -> None:
    """Launch the interactive REPL."""

    # ── Resolve config ───────────────────────────────────────────────────
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = api_key or minimax_key or anthropic_key or ""
    if not api_key:
        console.print("[err]Error:[/] Set MINIMAX_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    is_minimax = bool(minimax_key) and api_key == minimax_key
    provider_enum = LLMProvider(provider)
    model = model or os.environ.get("MINI_AGENT_MODEL") or (
        "MiniMax-M2.7" if is_minimax else "claude-sonnet-4-20250514"
    )
    api_base = api_base or os.environ.get("MINI_AGENT_API_BASE") or (
        "https://api.minimax.io" if is_minimax else "https://api.anthropic.com"
    )
    workspace = workspace or os.getcwd()

    llm_client = LLMClient(
        api_key=api_key, provider=provider_enum,
        api_base=api_base, model=model,
    )

    tools = [
        ReadTool(workspace_dir=workspace),
        WriteTool(workspace_dir=workspace),
        EditTool(workspace_dir=workspace),
        BashTool(workspace_dir=workspace),
        GlobTool(workspace_dir=workspace),
        GrepTool(workspace_dir=workspace),
    ]

    perm_mgr = PermissionManager()
    perm_cb = perm_mgr.check if enable_permissions else None

    default_prompt = system_prompt or (
        "You are a helpful coding assistant. You have access to tools for "
        "reading/writing files, running shell commands, and more. "
        "Work step by step and verify your changes."
    )

    tool_result_store = ToolResultStore(
        storage_dir=str(Path(workspace) / ".runtime" / "tool_results")
    )

    hooks = HookRegistry()
    dream = DreamConsolidator(
        memory_dir=str(Path(workspace) / ".claude" / "memory"),
        llm_client=llm_client,
    )
    dream.register(hooks)

    agent = Agent(
        llm_client=llm_client,
        system_prompt=default_prompt,
        tools=tools,
        max_steps=max_steps,
        tool_result_store=tool_result_store,
        project_dir=workspace,
        permission_callback=perm_cb,
        hooks=hooks,
    )

    # Load existing session
    if session_file and Path(session_file).exists():
        agent.messages = load_session(session_file)
        console.print(f"[info]Session loaded from {session_file}[/]")

    # ── Prompt session ───────────────────────────────────────────────────
    history_path = Path(workspace) / ".runtime" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_path)))

    kb = KeyBindings()

    @kb.add(Keys.Escape, Keys.Enter)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    # ── Banner ───────────────────────────────────────────────────────────
    console.print()
    console.print(f"  [bold cyan]mini-agent[/]  [dim]v0.1.0[/]")
    console.print(f"  [dim]{model}  ·  {Path(workspace).name}[/]")
    console.print()
    console.print(Rule(style="dim"))
    console.print(f"  [dim]Enter send · Esc+Enter newline · ^C stop · ^D quit · /help commands[/]")
    console.print(Rule(style="dim"))
    console.print()

    cancel_event = asyncio.Event()
    show_thinking = False

    # ── REPL loop ────────────────────────────────────────────────────────
    while True:
        # Build prompt
        token_est = estimate_tokens(agent.messages)

        try:
            user_input = await session.prompt_async(
                FormattedText([
                    ("", " "),
                    ("bold cyan", "› "),
                ]),
                multiline=False,
                key_bindings=kb,
            )
        except KeyboardInterrupt:
            continue
        except EOFError:
            # Ctrl+D → quit
            if len(agent.messages) > 1:
                auto_path = str(Path(workspace) / ".runtime" / "autosave_session.json")
                _save_session(agent.messages, auto_path)
                console.print(f"\n[info]Session saved → {auto_path}[/]")
            console.print("[dim]Bye![/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Slash commands ───────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                console.print()
                for c, d in [
                    ("/help",         "Show commands"),
                    ("/clear",        "Clear conversation"),
                    ("/compact",      "Compress context"),
                    ("/thinking",     "Toggle thinking display"),
                    ("/model [name]", "Show/change model"),
                    ("/history",      "Turns & token count"),
                    ("/cost",         "Token estimate"),
                    ("/save [path]",  "Save session"),
                    ("/load <path>",  "Load session"),
                    ("/exit",         "Quit"),
                ]:
                    console.print(f"  [dim]{c:18s}[/] {d}")
                console.print()
                continue

            if cmd in ("/exit", "/quit"):
                console.print("[dim]Bye![/]")
                break

            if cmd == "/clear":
                sys_msg = agent.messages[0] if agent.messages and agent.messages[0].role == "system" else None
                agent.messages = [sys_msg] if sys_msg else []
                console.print("[info]Cleared.[/]")
                continue

            if cmd == "/compact":
                before = estimate_tokens(agent.messages)
                agent.messages = await compact_messages(
                    agent.messages, agent.llm, token_threshold=0
                )
                after = estimate_tokens(agent.messages)
                console.print(f"[info]Compacted: {_format_tokens(before)} → {_format_tokens(after)}[/]")
                continue

            if cmd == "/thinking":
                show_thinking = not show_thinking
                state = "on" if show_thinking else "off"
                console.print(f"[info]Thinking display: {state}[/]")
                continue

            if cmd == "/model":
                if arg:
                    agent.llm = LLMClient(
                        api_key=api_key, provider=provider_enum,
                        api_base=api_base, model=arg,
                    )
                    model = arg
                    console.print(f"[info]Model → {arg}[/]")
                else:
                    console.print(f"[info]Model: {model}[/]")
                continue

            if cmd == "/history":
                turns = len([m for m in agent.messages if m.role in ("user", "assistant")])
                tokens = estimate_tokens(agent.messages)
                console.print(f"[info]{turns} turns · {_format_tokens(tokens)} tokens[/]")
                continue

            if cmd == "/cost":
                tokens = estimate_tokens(agent.messages)
                console.print(f"[info]{_format_tokens(tokens)} tokens[/]")
                continue

            if cmd == "/save":
                path = arg or "session.json"
                _save_session(agent.messages, path)
                console.print(f"[info]Saved → {path}[/]")
                continue

            if cmd == "/load":
                if not arg:
                    console.print("[err]Usage: /load <path>[/]")
                    continue
                if not Path(arg).exists():
                    console.print(f"[err]Not found: {arg}[/]")
                    continue
                agent.messages = load_session(arg)
                console.print(f"[info]Loaded ← {arg}[/]")
                continue

            console.print(f"[err]Unknown: {cmd}[/]  [dim](try /help)[/]")
            continue

        # ── Send to agent ────────────────────────────────────────────────
        console.print()  # blank line after user input
        agent.add_user_message(user_input)
        cancel_event.clear()
        start_time = time.monotonic()
        text_buf = ""

        try:
            async for event in agent.run_stream(cancel_event=cancel_event):
                if isinstance(event, TextChunk):
                    text_buf += event.content

                elif isinstance(event, ThinkingChunk):
                    _flush(text_buf)
                    text_buf = ""
                    if show_thinking:
                        console.print(f"  [thinking]··· {_truncate(event.content, 150)}[/]")
                    else:
                        console.print(f"  [thinking]···[/]")

                elif isinstance(event, ToolStart):
                    _flush(text_buf)
                    text_buf = ""
                    args_str = ", ".join(
                        f"{k}={_truncate(str(v), 50)}"
                        for k, v in event.arguments.items()
                    )
                    console.print(f"  [tool]▸ {event.tool_name}[/][dim]({args_str})[/]")

                elif isinstance(event, ToolEnd):
                    if event.success:
                        preview = _truncate(event.content, 100) if event.content else "ok"
                        console.print(f"  [tool.ok]✓ {event.tool_name}[/] [dim]→ {preview}[/]")
                    else:
                        console.print(f"  [tool.err]✗ {event.tool_name}[/] [dim]→ {event.error}[/]")

                elif isinstance(event, PermissionRequest):
                    _flush(text_buf)
                    text_buf = ""

                elif isinstance(event, AgentDone):
                    _flush(text_buf)
                    text_buf = ""
                    elapsed = time.monotonic() - start_time
                    console.print(f"  [done]{event.steps} steps · {elapsed:.1f}s[/]")
                    console.print()

                elif isinstance(event, AgentError):
                    _flush(text_buf)
                    text_buf = ""
                    console.print()
                    console.print(f"  [err]✗ {event.error}[/]")
                    console.print()

        except KeyboardInterrupt:
            _flush(text_buf)
            cancel_event.set()
            console.print()
            console.print(f"  [dim]Interrupted.[/]")
            console.print()


def _flush(text_buf: str) -> None:
    """Render accumulated text as a markdown panel."""
    if not text_buf:
        return
    console.print()
    console.print(
        Panel(
            Markdown(text_buf),
            border_style="dim",
            padding=(0, 1),
            expand=True,
        )
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Parse args and launch TUI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="Mini-Agent — interactive AI coding assistant",
    )
    parser.add_argument("--api-key", help="API key (or set ANTHROPIC_API_KEY / MINIMAX_API_KEY)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--workspace", "-w", help="Working directory (default: cwd)")
    parser.add_argument("--system-prompt", help="Custom system prompt text")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per turn")
    parser.add_argument("--session", help="Session file to load/resume")
    parser.add_argument("--no-permissions", action="store_true",
                        help="Disable permission prompts (auto-allow all tools)")
    args = parser.parse_args()

    asyncio.run(run_tui(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        workspace=args.workspace,
        system_prompt=args.system_prompt,
        max_steps=args.max_steps,
        session_file=args.session,
        enable_permissions=not args.no_permissions,
    ))
