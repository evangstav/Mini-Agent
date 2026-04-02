"""Fork commands: /fork, /forks."""

from . import REPLContext
from ..render.tables import forks_table


class ForkCommand:
    name = "/fork"
    aliases: list[str] = []
    description = "Fork conversation (/fork [N] — branch from N turns ago)"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        rewind = 0
        if arg:
            try:
                rewind = int(arg)
            except ValueError:
                ctx.console.print("[error]Usage: /fork [N] — N is number of turns to rewind[/]")
                return True

        forked = ctx.agent.fork(rewind=rewind)
        fork_id = ctx.next_fork_id
        ctx.next_fork_id += 1
        ctx.forks[fork_id] = forked
        ctx.agent = forked
        ctx.current_fork_id = fork_id

        turns = len([m for m in ctx.agent.messages if m.role in ("user", "assistant")])
        rewind_label = f", rewound {rewind} turns" if rewind else ""
        ctx.console.print(f"[success]Forked![/] Fork #{fork_id} ({turns} turns{rewind_label})")
        ctx.console.print("[info]Tip: Use /forks to list and switch forks.[/]")
        return True


class ForksCommand:
    name = "/forks"
    aliases: list[str] = []
    description = "List forks and switch (/forks, /forks <id>)"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        if arg:
            try:
                target_id = int(arg)
            except ValueError:
                ctx.console.print("[error]Usage: /forks [id][/]")
                return True
            if target_id not in ctx.forks:
                ctx.console.print(f"[error]No fork with id:[/] {target_id}")
                return True
            ctx.agent = ctx.forks[target_id]
            ctx.current_fork_id = target_id
            ctx.console.print(f"[success]Switched to fork[/] #{target_id}")
            return True

        table = forks_table(ctx.forks, ctx.current_fork_id)
        ctx.console.print(table)
        return True
