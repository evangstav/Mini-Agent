"""Event renderer — dispatches AgentEvent objects to Rich output."""

import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ...cost import calculate_cost, format_cost
from ...events import (
    AgentCancelled,
    AgentDone,
    AgentError,
    AgentEvent,
    PermissionRequest,
    PlanProposal,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s


class EventRenderer:
    """Renders AgentEvent objects to the terminal via Rich."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._in_text = False
        self._text_buffer = ""

    def render(self, event: AgentEvent) -> None:
        if isinstance(event, TextChunk):
            self._render_text_chunk(event)
        elif isinstance(event, ThinkingChunk):
            self._render_thinking(event)
        elif isinstance(event, ToolStart):
            self._render_tool_start(event)
        elif isinstance(event, ToolEnd):
            self._render_tool_end(event)
        elif isinstance(event, PlanProposal):
            self._render_plan(event)
        elif isinstance(event, AgentDone):
            pass  # handled by render_done
        elif isinstance(event, AgentError):
            self._render_error(event)
        elif isinstance(event, AgentCancelled):
            pass  # handled by render_cancelled
        elif isinstance(event, PermissionRequest):
            pass  # handled by callback

    def _render_text_chunk(self, event: TextChunk) -> None:
        if not self._in_text:
            self._console.print()
            self._in_text = True
        sys.stdout.write(event.content)
        sys.stdout.flush()
        self._text_buffer += event.content

    def _render_thinking(self, event: ThinkingChunk) -> None:
        self._flush_text()
        self._console.print(f"  [thinking]{_truncate(event.content, 200)}[/]")

    def _render_tool_start(self, event: ToolStart) -> None:
        self._flush_text()
        args_str = ", ".join(f"{k}={_truncate(str(v), 60)}" for k, v in event.arguments.items())
        self._console.print(f"\n  [bold blue]▶[/] [tool.name]{event.tool_name}[/]([tool.args]{args_str}[/])")

    def _render_tool_end(self, event: ToolEnd) -> None:
        if event.success:
            preview = _truncate(event.content, 200) if event.content else "(empty)"
            self._console.print(f"  [tool.success]✓[/] [tool.preview]{preview}[/]")
        else:
            self._console.print(f"  [tool.error]✗ {event.error or 'failed'}[/]")

    def _render_plan(self, event: PlanProposal) -> None:
        self._flush_text()
        table = Table(title=f"Proposed Plan ({len(event.proposed_calls)} actions)", show_header=True, show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Tool", style="tool.name")
        table.add_column("Arguments")

        for i, call in enumerate(event.proposed_calls, 1):
            args_lines = []
            for k, v in call["arguments"].items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                args_lines.append(f"[dim]{k}:[/] {v_str}")
            table.add_row(str(i), call["name"], "\n".join(args_lines))

        self._console.print()
        self._console.print(table)

    def _render_error(self, event: AgentError) -> None:
        self._flush_text()
        self._console.print(f"\n[error]Error:[/] {event.error}")

    def render_done(self, event: AgentDone, elapsed: float, model: str, token_usage=None) -> None:
        self._flush_text()
        parts = [f"{event.steps} steps", f"{elapsed:.1f}s"]
        if token_usage and token_usage.total_tokens > 0:
            cost = calculate_cost(token_usage, model)
            parts.append(format_cost(cost))
        self._console.print(f"\n[info]Done ({', '.join(parts)})[/]")

    def render_cancelled(self, event: AgentCancelled, elapsed: float) -> None:
        self._flush_text()
        self._console.print(f"\n[warning]Cancelled ({event.steps} steps, {elapsed:.1f}s)[/]")

    def flush_text(self) -> str:
        """Flush and return buffered text. Call at end of response."""
        text = self._text_buffer
        self._flush_text()
        return text

    def _flush_text(self) -> None:
        if self._in_text:
            print()  # newline after streamed text
            self._in_text = False
        self._text_buffer = ""
