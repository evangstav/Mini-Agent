"""Backward-compatibility shim — delegates to cli package."""

from .cli import main

__all__ = ["main"]
