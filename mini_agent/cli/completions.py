"""Tab completion for slash commands."""

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from .commands import SlashCommandRegistry


class SlashCommandCompleter(Completer):
    """Tab completion for slash commands."""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self._registry = registry

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        for name in self._registry.all_names():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text))
