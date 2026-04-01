"""TUI — Interactive terminal interface for Mini-Agent.

Provides a Claude-Code-like REPL with streaming output, tool call display,
permission prompts, slash commands, and session save/load.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from .agent import Agent
from .context import ToolResultStore, estimate_tokens
from .dream import DreamConsolidator
from .hooks import HookRegistry
from .events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)
from .llm import LLMClient
from .schema import LLMProvider, Message
from .tools.bash_tool import BashTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.git_tool import GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from .log import setup_logging
from .tools.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async
from .tools.web_fetch import WebFetchTool
from .tools.web_search import WebSearchTool

# ── ANSI helpers ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"
GRAY = "\033[90m"


def _styled(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


# ── Permission callback ──────────────────────────────────────────────────────

class PermissionManager:
    """Manages tool permission prompts (y/n/always)."""

    def __init__(self):
        self._always_allowed: set[str] = set()

    async def check(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name in self._always_allowed:
            return True

        # Show what's being requested
        args_preview = json.dumps(arguments, indent=2)
        if len(args_preview) > 300:
            args_preview = args_preview[:300] + "..."

        print(f"\n{_styled('Permission requested:', YELLOW, BOLD)} {_styled(tool_name, CYAN)}")
        print(f"{_styled('Arguments:', DIM)}")
        print(f"{GRAY}{args_preview}{RESET}")

        while True:
            try:
                answer = input(f"{_styled('[y]es / [n]o / [a]lways: ', YELLOW)}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                self._always_allowed.add(tool_name)
                return True


# ── Session persistence ───────────────────────────────────────────────────────

def save_session(messages: list[Message], path: str) -> None:
    """Save conversation to JSON."""
    data = [msg.model_dump() for msg in messages]
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_session(path: str) -> list[Message]:
    """Load conversation from JSON.

    Raises ValueError with a descriptive message if the file contains
    invalid JSON or messages that don't conform to the Message schema.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupt session file (invalid JSON): {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"Corrupt session file: expected a JSON array, got {type(raw).__name__}")

    messages: list[Message] = []
    for i, item in enumerate(raw):
        try:
            messages.append(Message(**item))
        except Exception as e:
            raise ValueError(f"Corrupt session file: invalid message at index {i}: {e}") from e
    return messages


# ── Cost estimation ───────────────────────────────────────────────────────────

def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ── Slash commands ────────────────────────────────────────────────────────────

SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation history (keep system prompt)",
    "/compact": "Force context compaction",
    "/model": "Show or change model",
    "/history": "Show conversation turn count and token estimate",
    "/cost": "Show estimated token usage",
    "/save": "Save session to file (/save [path])",
    "/load": "Load session from file (/load <path>)",
    "/exit": "Exit the REPL",
}


def _print_help() -> None:
    print(f"\n{_styled('Mini-Agent Commands', BOLD, CYAN)}")
    print(f"{_styled('─' * 40, DIM)}")
    for cmd, desc in SLASH_COMMANDS.items():
        print(f"  {_styled(cmd, GREEN):24s} {desc}")
    print()


# ── Rendering helpers ────────────────────────────────────────────────────────

def _render_tool_start(event: ToolStart) -> None:
    args_str = ", ".join(f"{k}={_truncate(str(v), 60)}" for k, v in event.arguments.items())
    print(f"\n  {_styled('▶', BLUE)} {_styled(event.tool_name, CYAN, BOLD)}({_styled(args_str, DIM)})")


def _render_tool_end(event: ToolEnd) -> None:
    if event.success:
        preview = _truncate(event.content, 200) if event.content else "(empty)"
        print(f"  {_styled('✓', GREEN)} {_styled(preview, DIM)}")
    else:
        print(f"  {_styled('✗', RED)} {_styled(event.error or 'failed', RED)}")


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", "↵ ")
    return s[:n] + "…" if len(s) > n else s


# ── Main REPL ─────────────────────────────────────────────────────────────────

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
    verbose: bool = False,
) -> None:
    """Launch the interactive TUI REPL."""
    setup_logging(verbose=verbose)

    # ── Resolve config ────────────────────────────────────────────────────
    # Auto-detect provider from available API keys
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = api_key or minimax_key or anthropic_key or ""
    if not api_key:
        logger.error("No API key configured")
        print(f"{_styled('Error:', RED, BOLD)} Set MINIMAX_API_KEY or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # Default to MiniMax if MINIMAX_API_KEY is set, otherwise Anthropic
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
        api_key=api_key,
        provider=provider_enum,
        api_base=api_base,
        model=model,
    )

    tools = [
        ReadTool(workspace_dir=workspace),
        WriteTool(workspace_dir=workspace),
        EditTool(workspace_dir=workspace),
        BashTool(workspace_dir=workspace),
        GitStatusTool(workspace_dir=workspace),
        GitDiffTool(workspace_dir=workspace),
        GitCommitTool(workspace_dir=workspace),
        GitLogTool(workspace_dir=workspace),
        GitBranchTool(workspace_dir=workspace),
        WebSearchTool(),
        WebFetchTool(),
    ]

    # Load MCP tools from config
    mcp_config = str(Path(workspace) / "mcp.json")
    mcp_tools = await load_mcp_tools_async(mcp_config)
    tools.extend(mcp_tools)

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

    # Set up hooks with dream consolidator for persistent memory
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

    # Load existing session if requested
    if session_file and Path(session_file).exists():
        agent.messages = load_session(session_file)
        print(f"{_styled('Session loaded from', DIM)} {session_file}")

    # ── Prompt session ────────────────────────────────────────────────────
    history_path = Path(workspace) / ".runtime" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_path)))

    # Key bindings for multi-line input (Alt+Enter or Escape then Enter to submit)
    kb = KeyBindings()

    @kb.add(Keys.Escape, Keys.Enter)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    logger.info("TUI started: model=%s provider=%s workspace=%s", model, provider, workspace)

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n{_styled('Mini-Agent', BOLD, CYAN)} {_styled('v0.1.0', DIM)}")
    print(f"{_styled('Model:', DIM)} {model}  {_styled('Provider:', DIM)} {provider}")
    print(f"{_styled('Workspace:', DIM)} {workspace}")
    print(f"{_styled('Type /help for commands. Ctrl+C to cancel. Ctrl+D to exit.', DIM)}\n")

    cancel_event = asyncio.Event()

    # ── REPL loop ─────────────────────────────────────────────────────────
    while True:
        # Build prompt with token count
        token_est = estimate_tokens(agent.messages)
        prompt_text = FormattedText([
            ("class:gray", f"[{_format_tokens(token_est)} tokens] "),
            ("class:prompt", "› "),
        ])

        try:
            user_input = await session.prompt_async(
                prompt_text,
                multiline=False,
                key_bindings=kb,
            )
        except KeyboardInterrupt:
            continue
        except EOFError:
            # Auto-save session before exit
            auto_save_path = str(Path(workspace) / ".runtime" / "autosave_session.json")
            if len(agent.messages) > 1:  # More than just the system prompt
                save_session(agent.messages, auto_save_path)
                print(f"\n{_styled('Session auto-saved to', DIM)} {auto_save_path}")
            print(f"{_styled('Goodbye!', DIM)}")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Handle slash commands ─────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                _print_help()
                continue

            if cmd == "/exit" or cmd == "/quit":
                print(f"{_styled('Goodbye!', DIM)}")
                break

            if cmd == "/clear":
                system_msg = agent.messages[0] if agent.messages and agent.messages[0].role == "system" else None
                agent.messages = [system_msg] if system_msg else []
                print(f"{_styled('Conversation cleared.', GREEN)}")
                continue

            if cmd == "/compact":
                before = estimate_tokens(agent.messages)
                from .context import compact_messages
                agent.messages = await compact_messages(
                    agent.messages, agent.llm, token_threshold=0
                )
                after = estimate_tokens(agent.messages)
                print(f"{_styled('Compacted:', GREEN)} {_format_tokens(before)} → {_format_tokens(after)} tokens")
                continue

            if cmd == "/model":
                if arg:
                    agent.llm = LLMClient(
                        api_key=api_key,
                        provider=provider_enum,
                        api_base=api_base,
                        model=arg,
                    )
                    model = arg
                    print(f"{_styled('Model changed to:', GREEN)} {arg}")
                else:
                    print(f"{_styled('Current model:', DIM)} {model}")
                continue

            if cmd == "/history":
                turns = len([m for m in agent.messages if m.role in ("user", "assistant")])
                tokens = estimate_tokens(agent.messages)
                print(f"{_styled('Turns:', DIM)} {turns}  {_styled('Est. tokens:', DIM)} {_format_tokens(tokens)}")
                continue

            if cmd == "/cost":
                tokens = estimate_tokens(agent.messages)
                print(f"{_styled('Estimated context:', DIM)} {_format_tokens(tokens)} tokens")
                continue

            if cmd == "/save":
                save_path = arg or "session.json"
                save_session(agent.messages, save_path)
                print(f"{_styled('Session saved to', GREEN)} {save_path}")
                continue

            if cmd == "/load":
                if not arg:
                    print(f"{_styled('Usage: /load <path>', RED)}")
                    continue
                if not Path(arg).exists():
                    print(f"{_styled('File not found:', RED)} {arg}")
                    continue
                try:
                    agent.messages = load_session(arg)
                except ValueError as e:
                    print(f"{_styled('Error loading session:', RED, BOLD)} {e}")
                    continue
                print(f"{_styled('Session loaded from', GREEN)} {arg}")
                continue

            print(f"{_styled('Unknown command:', RED)} {cmd}. Type /help for commands.")
            continue

        # ── Send to agent ─────────────────────────────────────────────────
        agent.add_user_message(user_input)
        cancel_event.clear()
        start_time = time.monotonic()
        in_text = False

        try:
            async for event in agent.run_stream(cancel_event=cancel_event):
                if isinstance(event, TextChunk):
                    if not in_text:
                        print()  # blank line before response
                        in_text = True
                    sys.stdout.write(event.content)
                    sys.stdout.flush()

                elif isinstance(event, ThinkingChunk):
                    # Show thinking in dim italic
                    print(f"\n{_styled('thinking:', DIM, ITALIC)} {_truncate(event.content, 200)}")

                elif isinstance(event, ToolStart):
                    if in_text:
                        print()  # end text block
                        in_text = False
                    _render_tool_start(event)

                elif isinstance(event, ToolEnd):
                    _render_tool_end(event)

                elif isinstance(event, PermissionRequest):
                    pass  # handled by the callback

                elif isinstance(event, AgentCancelled):
                    if in_text:
                        print()
                    elapsed = time.monotonic() - start_time
                    print(f"\n{_styled(f'Cancelled ({event.steps} steps, {elapsed:.1f}s)', YELLOW)}")

                elif isinstance(event, AgentDone):
                    if in_text:
                        print()
                    elapsed = time.monotonic() - start_time
                    print(f"\n{_styled(f'Done ({event.steps} steps, {elapsed:.1f}s)', DIM)}")

                elif isinstance(event, AgentError):
                    if in_text:
                        print()
                    print(f"\n{_styled('Error:', RED, BOLD)} {event.error}")

        except KeyboardInterrupt:
            cancel_event.set()
            print(f"\n{_styled('Cancelled.', YELLOW)}")

    # Session truly ending — emit SESSION_END for dream consolidation
    await agent.end_session()
    await cleanup_mcp_connections()


# ── CLI entry point ───────────────────────────────────────────────────────────

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
    parser.add_argument("--model", help="Model name (default: MiniMax-M2.7 or claude-sonnet-4-20250514)")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--workspace", "-w", help="Working directory (default: cwd)")
    parser.add_argument("--system-prompt", help="Custom system prompt text")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per turn")
    parser.add_argument("--session", help="Session file to load/resume")
    parser.add_argument("--no-permissions", action="store_true",
                        help="Disable permission prompts (auto-allow all tools)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging to stderr")
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
        verbose=args.verbose,
    ))
