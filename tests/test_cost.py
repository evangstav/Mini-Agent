"""Tests for cost estimation module."""

from mini_agent.cost import calculate_cost, format_cost, get_pricing
from mini_agent.schema import TokenUsage


class TestGetPricing:
    def test_known_model(self):
        inp, out = get_pricing("claude-sonnet-4-20250514")
        assert inp == 3.0
        assert out == 15.0

    def test_prefix_match(self):
        inp, out = get_pricing("claude-sonnet-4-future")
        assert inp == 3.0
        assert out == 15.0

    def test_unknown_model_returns_default(self):
        inp, out = get_pricing("unknown-model-xyz")
        assert inp == 3.0  # default
        assert out == 15.0  # default

    def test_minimax_model(self):
        inp, out = get_pricing("MiniMax-M2.7")
        assert inp == 1.0
        assert out == 4.0


class TestCalculateCost:
    def test_zero_usage(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert calculate_cost(usage, "claude-sonnet-4-20250514") == 0.0

    def test_known_model_cost(self):
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=100_000, total_tokens=1_100_000)
        cost = calculate_cost(usage, "claude-sonnet-4-20250514")
        # 1M * 3.0/1M + 100K * 15.0/1M = 3.0 + 1.5 = 4.5
        assert cost == 4.5


class TestFormatCost:
    def test_tiny_cost(self):
        assert format_cost(0.0012) == "$0.0012"

    def test_small_cost(self):
        assert format_cost(0.15) == "$0.15"

    def test_large_cost(self):
        assert format_cost(1.23) == "$1.23"

    def test_zero(self):
        assert format_cost(0.0) == "$0.0000"
