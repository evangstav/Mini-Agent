"""Help command."""

from . import REPLContext
from ..render.tables import help_table


class HelpCommand:
    name = "/help"
    aliases: list[str] = []
    description = "Show available commands"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        from . import SlashCommandRegistry
        # Access the registry from the REPL context
        registry = getattr(ctx, "_registry", None)
        if registry:
            table = help_table(registry.all_descriptions())
            ctx.console.print(table)
        return True
