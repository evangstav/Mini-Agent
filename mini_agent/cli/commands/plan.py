"""Plan command — propose changes without executing."""

import time

from ...events import (
    AgentDone,
    AgentError,
    PlanProposal,
    TextChunk,
    ThinkingChunk,
)

from . import REPLContext


class PlanCommand:
    name = "/plan"
    aliases: list[str] = []
    description = "Propose changes without executing (/plan <instruction>)"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        if not arg:
            ctx.console.print("[error]Usage: /plan <instruction>[/]")
            return True

        ctx.agent.add_user_message(arg)
        ctx.cancel_event.clear()
        start_time = time.monotonic()
        plan_event = None

        try:
            async for event in ctx.agent.run_stream(
                cancel_event=ctx.cancel_event, plan_mode=True
            ):
                if isinstance(event, (TextChunk, ThinkingChunk, PlanProposal)):
                    ctx.renderer.render(event)
                    if isinstance(event, PlanProposal):
                        plan_event = event
                elif isinstance(event, AgentDone):
                    ctx.renderer.flush_text()
                    if not plan_event:
                        elapsed = time.monotonic() - start_time
                        ctx.console.print(f"\n[info]No actions proposed ({event.steps} steps, {elapsed:.1f}s)[/]")
                elif isinstance(event, AgentError):
                    ctx.renderer.render(event)
        except KeyboardInterrupt:
            ctx.cancel_event.set()
            ctx.console.print("\n[warning]Cancelled.[/]")
            return True

        if plan_event:
            while True:
                try:
                    answer = input("[bold yellow][a]pprove / [r]eject: [/]").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "r"
                    break
                if answer in ("a", "approve", "r", "reject"):
                    break

            if answer in ("a", "approve"):
                ctx.console.print("[success]Executing plan...[/]")
                try:
                    async for event in ctx.agent.execute_plan(cancel_event=ctx.cancel_event):
                        ctx.renderer.render(event)
                        if isinstance(event, AgentDone):
                            elapsed = time.monotonic() - start_time
                            ctx.renderer.render_done(event, elapsed, ctx.model, ctx.agent.token_usage)
                except KeyboardInterrupt:
                    ctx.cancel_event.set()
                    ctx.console.print("\n[warning]Cancelled.[/]")
            else:
                await ctx.agent.reject_plan()
                ctx.console.print("[warning]Plan rejected.[/]")

        return True


class ExitCommand:
    name = "/exit"
    aliases = ["/quit"]
    description = "Exit the REPL"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        ctx.console.print("[info]Goodbye![/]")
        return False
