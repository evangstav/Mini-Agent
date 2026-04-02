"""LLM clients package supporting both Anthropic and OpenAI protocols."""

import os

from .anthropic_client import AnthropicClient
from .base import LLMClientBase
from .llm_wrapper import LLMClient
from .openai_client import OpenAIClient
from ..schema import LLMProvider


def auto_detect_provider() -> tuple[str, str, str, LLMProvider]:
    """Auto-detect API key, model, base URL, and provider from environment.

    Returns: (api_key, model, api_base, provider_enum)
    Raises ValueError if no API key found.
    """
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = minimax_key or anthropic_key

    if not api_key:
        raise ValueError("No API key found. Set MINIMAX_API_KEY or ANTHROPIC_API_KEY.")

    is_minimax = bool(minimax_key) and api_key == minimax_key
    provider = LLMProvider.ANTHROPIC  # Both use Anthropic protocol
    model = "MiniMax-M2.7" if is_minimax else "claude-sonnet-4-20250514"
    api_base = "https://api.minimax.io" if is_minimax else "https://api.anthropic.com"

    return api_key, model, api_base, provider


__all__ = [
    "LLMClientBase",
    "AnthropicClient",
    "OpenAIClient",
    "LLMClient",
    "auto_detect_provider",
]
