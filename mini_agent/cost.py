"""Per-session API cost estimation from real token counts.

Provides a configurable per-model pricing table and cost calculation
using cumulative input/output tokens tracked by the Agent.
"""

from __future__ import annotations

from .schema import TokenUsage

# Pricing: (input_cost_per_million, output_cost_per_million) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic models
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-3-5-20241022": (0.80, 4.0),
    # MiniMax models
    "MiniMax-M1": (0.50, 2.0),
    "MiniMax-M2.5": (1.0, 4.0),
    "MiniMax-M2.7": (1.0, 4.0),
}

# Fallback for unrecognized models
_DEFAULT_PRICING = (3.0, 15.0)


def get_pricing(model: str) -> tuple[float, float]:
    """Return (input_$/M, output_$/M) for a model, with prefix matching fallback."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Prefix match: "claude-sonnet-4-xxx" → "claude-sonnet-4-20250514"
    for key, pricing in MODEL_PRICING.items():
        if model.startswith(key.rsplit("-", 1)[0]):
            return pricing
    return _DEFAULT_PRICING


def calculate_cost(usage: TokenUsage, model: str) -> float:
    """Calculate session cost in USD from token usage and model pricing."""
    input_rate, output_rate = get_pricing(model)
    return (
        usage.prompt_tokens * input_rate / 1_000_000
        + usage.completion_tokens * output_rate / 1_000_000
    )


def format_cost(cost_usd: float) -> str:
    """Format cost for display: $0.0012, $0.15, $1.23."""
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    if cost_usd < 1.0:
        return f"${cost_usd:.2f}"
    return f"${cost_usd:.2f}"
