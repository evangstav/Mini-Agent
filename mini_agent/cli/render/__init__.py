"""Render package — Rich-based terminal output."""

from .console import console, THEME
from .events import EventRenderer

__all__ = ["console", "THEME", "EventRenderer"]
