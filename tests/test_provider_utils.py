from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)
from dr_llm.generation.models import TokenUsage


def test_token_usage_defaults_total() -> None:
    usage = TokenUsage.from_raw(
        prompt_tokens=10, completion_tokens=5, total_tokens=None
    )
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.reasoning_tokens == 0


def test_token_usage_includes_reasoning_tokens() -> None:
    usage = TokenUsage.from_raw(
        prompt_tokens=10, completion_tokens=5, total_tokens=20, reasoning_tokens=7
    )
    assert usage.reasoning_tokens == 7


def test_parse_reasoning_tokens_nested_details() -> None:
    usage_raw = {"completion_tokens_details": {"reasoning_tokens": 13}}
    assert parse_reasoning_tokens(usage_raw) == 13


def test_parse_reasoning_tokens_output_tokens_details() -> None:
    usage_raw = {"output_tokens_details": {"reasoning_tokens": 7}}
    assert parse_reasoning_tokens(usage_raw) == 7


def test_parse_reasoning_tokens_missing_returns_zero() -> None:
    assert parse_reasoning_tokens({"completion_tokens": 42}) == 0


def test_parse_reasoning_fields() -> None:
    reasoning, reasoning_details = parse_reasoning(
        {
            "reasoning": "thought process",
            "reasoning_details": [{"type": "reasoning.text", "text": "step"}],
        }
    )
    assert reasoning == "thought process"
    assert reasoning_details == [{"type": "reasoning.text", "text": "step"}]


def test_parse_cost_info() -> None:
    cost = parse_cost_info(
        {
            "usage": {
                "cost": 0.0123,
                "prompt_cost": 0.004,
                "completion_cost": 0.0083,
                "currency": "USD",
            }
        }
    )
    assert cost is not None
    assert cost.total_cost_usd == 0.0123
    assert cost.prompt_cost_usd == 0.004
    assert cost.completion_cost_usd == 0.0083
    assert cost.currency == "USD"


def test_parse_cost_info_missing_keys_returns_none() -> None:
    assert parse_cost_info({"usage": {"prompt_tokens": 1}}) is None


def test_parse_cost_info_ignores_non_string_currency_values() -> None:
    cost = parse_cost_info(
        {
            "usage": {
                "cost": 0.25,
                "currency": 0,
            }
        }
    )
    assert cost is not None
    assert cost.currency == "USD"
