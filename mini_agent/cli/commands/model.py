"""Model command."""

from ...llm import LLMClient

from . import REPLContext


class ModelCommand:
    name = "/model"
    aliases: list[str] = []
    description = "Show or change model"

    async def execute(self, arg: str, ctx: REPLContext) -> bool:
        if arg:
            ctx.agent.llm = LLMClient(
                api_key=ctx.api_key,
                provider=ctx.provider_enum,
                api_base=ctx.api_base,
                model=arg,
            )
            ctx.model = arg
            if ctx.agent_tool:
                ctx.agent_tool._llm_client = ctx.agent.llm
            if ctx.dream:
                ctx.dream.llm_client = ctx.agent.llm
            ctx.console.print(f"[success]Model changed to:[/] {arg}")
        else:
            ctx.console.print(f"[info]Current model:[/] {ctx.model}")
        return True
