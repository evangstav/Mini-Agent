"""CLI entry point with subcommands."""

import argparse
import asyncio
import os
from pathlib import Path

from .. import __version__
from .render.console import console


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="Mini-Agent -- interactive AI coding assistant",
    )

    # Shared flags
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging to stderr")

    sub = parser.add_subparsers(dest="command")

    # chat (default)
    sub.add_parser("chat", help="Launch interactive REPL (default)")

    # config
    config_parser = sub.add_parser("config", help="Configuration management")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="Show resolved configuration")
    config_sub.add_parser("init", help="Create template .mini-agent.toml")

    # session
    session_parser = sub.add_parser("session", help="Session management")
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_sub.add_parser("list", help="List saved sessions")
    load_parser = session_sub.add_parser("load", help="Load and resume a session")
    load_parser.add_argument("path", help="Path to session file")

    # run (non-interactive)
    run_parser = sub.add_parser("run", help="Run a single prompt non-interactively")
    run_parser.add_argument("prompt", nargs="?", help="The prompt to run (or pipe via stdin)")
    run_parser.add_argument("--print", "-p", dest="print_output", action="store_true",
                            help="Print only the final response (no tool output)")
    run_parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps")

    # version
    sub.add_parser("version", help="Show version")

    # bench
    bench_parser = sub.add_parser("bench", help="Run benchmarks")
    bench_sub = bench_parser.add_subparsers(dest="bench_command")

    swe_parser = bench_sub.add_parser("swebench", help="Run SWE-bench Verified")
    swe_parser.add_argument("--slice", default=None, help="Instance range, e.g. '0:5' (default: all)")
    swe_parser.add_argument("--subset", default="verified", choices=["verified", "lite", "full"],
                            help="Dataset subset (default: verified)")
    swe_parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per instance")
    swe_parser.add_argument("--attempts", type=int, default=1,
                            help="Best-of-N attempts per instance (default: 1, try 3 for +15-30%%)")
    swe_parser.add_argument("--output", default="predictions.jsonl", help="Output file")

    he_parser = bench_sub.add_parser("humaneval", help="Run HumanEval+")
    he_parser.add_argument("--slice", default=None, help="Problem range, e.g. '0:5'")
    he_parser.add_argument("--output", default="humaneval_results.jsonl", help="Output file")

    return parser


def _cmd_version(args: argparse.Namespace) -> None:
    console.print(f"[heading]Mini-Agent[/] v{__version__}")


def _cmd_config_show(args: argparse.Namespace) -> None:
    from ..config import load_config
    from rich.table import Table

    from .setup import _detect_project_root
    workspace = args.workspace or _detect_project_root(os.getcwd())
    cfg = load_config(project_dir=workspace)

    table = Table(title="Resolved Configuration", show_header=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("model", cfg.model or "(default)")
    table.add_row("provider", cfg.provider or "anthropic")
    table.add_row("api_base", cfg.api_base or "(default)")
    table.add_row("max_steps", str(cfg.max_steps) if cfg.max_steps else "100")
    table.add_row("permissions", str(cfg.permissions) if cfg.permissions is not None else "true")
    table.add_row("tools", ", ".join(cfg.tools) if cfg.tools else "(all)")
    table.add_row("workspace", workspace)

    console.print(table)


def _cmd_config_init(args: argparse.Namespace) -> None:
    from .setup import _detect_project_root
    workspace = args.workspace or _detect_project_root(os.getcwd())
    path = Path(workspace) / ".mini-agent.toml"

    if path.exists():
        console.print(f"[warning]Config already exists:[/] {path}")
        return

    template = '''\
# Mini-Agent project configuration
# See: https://github.com/evangstav/Mini-Agent

[agent]
# model = "claude-sonnet-4-20250514"
# provider = "anthropic"
# api_base = "https://api.anthropic.com"
# max_steps = 100
# permissions = true
# tools = ["read_file", "write_file", "edit_file", "bash", "glob", "grep"]

# [[rules]]
# tool = "bash"
# action = "allow"
# args = { command = "git *" }
'''
    path.write_text(template, encoding="utf-8")
    console.print(f"[success]Created[/] {path}")


def _cmd_session_list(args: argparse.Namespace) -> None:
    from .setup import _detect_project_root
    workspace = args.workspace or _detect_project_root(os.getcwd())
    runtime = Path(workspace) / ".runtime"

    if not runtime.exists():
        console.print("[info]No sessions found.[/]")
        return

    sessions = sorted(runtime.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        console.print("[info]No sessions found.[/]")
        return

    from rich.table import Table
    table = Table(title="Saved Sessions", show_header=True)
    table.add_column("File")
    table.add_column("Size", justify="right")
    table.add_column("Modified")

    import time
    for s in sessions:
        stat = s.stat()
        size = f"{stat.st_size / 1024:.1f}K"
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        table.add_row(s.name, size, mtime)

    console.print(table)


def _cmd_session_load(args: argparse.Namespace) -> None:
    # Launch REPL with session loaded
    asyncio.run(_async_repl(args, session_file=args.path))


async def _async_repl(args: argparse.Namespace, session_file: str | None = None) -> None:
    from .setup import build_repl_context
    from .repl import launch_repl

    ctx = await build_repl_context(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        workspace=args.workspace,
        system_prompt=args.system_prompt,
        max_steps=args.max_steps,
        session_file=session_file or args.session,
        enable_permissions=not args.no_permissions,
        verbose=args.verbose,
    )
    await launch_repl(ctx)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a single prompt non-interactively."""
    import sys

    # Get prompt from arg or stdin
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            console.print("[error]Provide a prompt as argument or pipe via stdin.[/]")
            console.print("[info]Usage: mini-agent run 'fix the bug' or echo 'fix it' | mini-agent run[/]")
            sys.exit(1)

    async def _run():
        from .setup import build_repl_context
        ctx = await build_repl_context(
            api_key=args.api_key,
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            workspace=args.workspace,
            system_prompt=args.system_prompt,
            max_steps=getattr(args, "max_steps", 50),
            enable_permissions=not args.no_permissions,
            verbose=args.verbose,
        )
        ctx.agent.add_user_message(prompt)

        if getattr(args, "print_output", False):
            # Print only the final text response
            result = await ctx.agent.run()
            print(result)
        else:
            # Stream all events with rendering
            from ..events import AgentDone, AgentError, AgentCancelled
            import time
            start = time.monotonic()
            async for event in ctx.agent.run_stream():
                ctx.renderer.render(event)
                if isinstance(event, AgentDone):
                    elapsed = time.monotonic() - start
                    ctx.renderer.render_done(event, elapsed, ctx.model, ctx.agent.token_usage)
                elif isinstance(event, AgentError):
                    sys.exit(1)

        await ctx.agent.end_session()
        from ..tools.mcp_loader import cleanup_mcp_connections
        await cleanup_mcp_connections()

    asyncio.run(_run())


def _cmd_bench(args: argparse.Namespace) -> None:
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    bench = args.bench_command

    if bench == "swebench":
        from ..benchmarks.swebench_runner import SWEBenchRunner
        runner = SWEBenchRunner(
            model=args.model,
            provider=args.provider if args.provider != "anthropic" else None,
            api_key=args.api_key,
            api_base=args.api_base,
            max_steps=getattr(args, "max_steps", 30),
        )
        out = asyncio.run(runner.run_dataset(
            subset=args.subset,
            slice_range=args.slice,
            output_path=args.output,
            attempts=getattr(args, "attempts", 1),
        ))
        console.print(f"[success]Predictions written to[/] {out}")
        console.print("[info]Evaluate with: sb-cli submit or swebench.harness.run_evaluation[/]")

    elif bench == "humaneval":
        from ..benchmarks.humaneval_runner import HumanEvalRunner
        runner = HumanEvalRunner(
            model=args.model,
            provider=args.provider if args.provider != "anthropic" else None,
            api_key=args.api_key,
            api_base=args.api_base,
        )
        out = asyncio.run(runner.run_all(
            slice_range=args.slice,
            output_path=args.output,
        ))
        console.print(f"[success]Results written to[/] {out}")
        console.print("[info]Evaluate with: evalplus.evaluate --dataset humaneval --samples " + str(out) + "[/]")

    else:
        console.print("[info]Available benchmarks: swebench, humaneval[/]")


def main() -> None:
    """Parse args and dispatch."""
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command

    if cmd == "version":
        _cmd_version(args)
    elif cmd == "config":
        if args.config_command == "show":
            _cmd_config_show(args)
        elif args.config_command == "init":
            _cmd_config_init(args)
        else:
            _cmd_config_show(args)
    elif cmd == "session":
        if args.session_command == "list":
            _cmd_session_list(args)
        elif args.session_command == "load":
            _cmd_session_load(args)
        else:
            _cmd_session_list(args)
    elif cmd == "run":
        _cmd_run(args)
    elif cmd == "bench":
        _cmd_bench(args)
    else:
        # Default: launch REPL (no subcommand or "chat")
        asyncio.run(_async_repl(args))
