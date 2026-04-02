"""Bottom toolbar for the REPL prompt."""

from prompt_toolkit.formatted_text import HTML

from ..context import estimate_tokens


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def make_bottom_toolbar(ctx):
    """Return a callable for prompt_toolkit's bottom_toolbar."""
    def toolbar():
        agent = ctx.agent
        api_total = agent.token_usage.total_tokens
        if api_total > 0:
            from ..cost import calculate_cost, format_cost
            cost = calculate_cost(agent.token_usage, ctx.model)
            token_label = f"{_format_tokens(api_total)} tokens | {format_cost(cost)}"
        else:
            token_label = f"~{_format_tokens(estimate_tokens(agent.messages))} tokens"

        return HTML(
            f" <b>{ctx.model}</b>"
            f" | {token_label}"
            f" | {ctx.workspace}"
        )
    return toolbar
