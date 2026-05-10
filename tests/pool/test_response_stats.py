"""Unit tests for response stats parsing."""

from __future__ import annotations

from dr_llm.llm import ProviderName
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.response_stats import (
    parse_response_stats,
    sample_response_stats,
)


def test_parse_full_response() -> None:
    blob = {
        "text": "hello",
        "latency_ms": 250,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 10,
        },
        "cost": {
            "total_cost_usd": 0.003,
            "prompt_cost_usd": 0.001,
            "completion_cost_usd": 0.002,
        },
        "provider": ProviderName.ANTHROPIC,
        "model": "claude-3",
    }
    stats = parse_response_stats(blob, finish_reason="stop", attempt_count=1)

    assert stats.latency_ms == 250
    assert stats.total_cost_usd == 0.003
    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 50
    assert stats.reasoning_tokens == 10
    assert stats.total_tokens == 150
    assert stats.finish_reason == "stop"
    assert stats.attempt_count == 1


def test_parse_none_response() -> None:
    stats = parse_response_stats(None, finish_reason="error", attempt_count=3)

    assert stats.latency_ms is None
    assert stats.total_cost_usd is None
    assert stats.prompt_tokens is None
    assert stats.finish_reason == "error"
    assert stats.attempt_count == 3


def test_parse_missing_cost() -> None:
    blob = {
        "text": "hello",
        "latency_ms": 100,
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "reasoning_tokens": 0,
        },
    }
    stats = parse_response_stats(blob, finish_reason="stop", attempt_count=1)

    assert stats.latency_ms == 100
    assert stats.total_cost_usd is None
    assert stats.prompt_tokens == 50


def test_parse_malformed_usage_and_cost_as_absent() -> None:
    stats = parse_response_stats(
        {
            "latency_ms": 100,
            "usage": "n/a",
            "cost": 1,
        },
        finish_reason="stop",
        attempt_count=1,
    )

    assert stats.latency_ms == 100
    assert stats.total_cost_usd is None
    assert stats.prompt_tokens is None
    assert stats.completion_tokens is None
    assert stats.reasoning_tokens is None
    assert stats.total_tokens is None
    assert stats.finish_reason == "stop"
    assert stats.attempt_count == 1


def test_sample_response_stats_delegates() -> None:
    sample = PoolSample(
        key_values={"x": "a"},
        response={
            "latency_ms": 300,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        finish_reason="stop",
        attempt_count=2,
    )
    stats = sample_response_stats(sample)

    assert stats.latency_ms == 300
    assert stats.prompt_tokens == 10
    assert stats.finish_reason == "stop"
    assert stats.attempt_count == 2
