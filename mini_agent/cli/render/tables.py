"""Rich tables for info commands."""

from rich.table import Table

from ...cost import calculate_cost, format_cost
from ...schema import TokenUsage


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def cost_table(usage: TokenUsage, model: str) -> Table:
    """Build a Rich table showing token usage and cost."""
    table = Table(title="Session Cost", show_header=False, show_edge=False, pad_edge=False)
    table.add_column("Label", style="info.label")
    table.add_column("Value", style="info.value")

    if usage.total_tokens > 0:
        cost = calculate_cost(usage, model)
        table.add_row("Cost", format_cost(cost))
        table.add_row("Input", f"{_format_tokens(usage.prompt_tokens)} tokens")
        table.add_row("Output", f"{_format_tokens(usage.completion_tokens)} tokens")
        table.add_row("Total", f"{_format_tokens(usage.total_tokens)} tokens")
        table.add_row("Model", model)
    else:
        table.add_row("Status", "No API calls yet")

    return table


def forks_table(forks: dict, current_fork_id: int) -> Table:
    """Build a Rich table showing conversation forks."""
    table = Table(title="Conversation Forks", show_header=True)
    table.add_column("ID", style="bold cyan")
    table.add_column("Label")
    table.add_column("Turns", justify="right")
    table.add_column("Active")

    for fid, fagent in forks.items():
        turns = len([m for m in fagent.messages if m.role in ("user", "assistant")])
        label = "main" if fid == 0 else "fork"
        active = "[bold green]<--[/]" if fid == current_fork_id else ""
        table.add_row(f"#{fid}", label, str(turns), active)

    return table


def help_table(commands: dict[str, str]) -> Table:
    """Build a Rich table for /help."""
    table = Table(title="Mini-Agent Commands", show_header=False, show_edge=False, pad_edge=False)
    table.add_column("Command", style="command", min_width=16)
    table.add_column("Description", style="command.desc")

    for cmd, desc in commands.items():
        table.add_row(cmd, desc)

    return table
