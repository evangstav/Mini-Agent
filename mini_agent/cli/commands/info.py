"""Info commands: /history, /cost."""

from ...context import estimate_tokens

from . import REPLContext
from ..render.tables import _format_tokens, cost_table


class HistoryCommand:
    name = "/history"
    aliases: list[str] = []
    description = "Show conversation turn count and token estimate"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        turns = len([m for m in ctx.agent.messages if m.role in ("user", "assistant")])
        tokens = estimate_tokens(ctx.agent.messages)
        ctx.console.print(f"[info]Turns:[/] {turns}  [info]Est. tokens:[/] {_format_tokens(tokens)}")
        return True


class CostCommand:
    name = "/cost"
    aliases: list[str] = []
    description = "Show session token usage and estimated cost"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        table = cost_table(ctx.agent.token_usage, ctx.model)
        ctx.console.print(table)
        return True
