from __future__ import annotations

import pytest

from dr_llm.pool.call_stats import CallStats


def _full_response() -> dict:
    return {
        "text": "hello",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 20,
        },
        "cost": {
            "total_cost_usd": 0.005,
            "prompt_cost_usd": 0.002,
            "completion_cost_usd": 0.003,
        },
        "latency_ms": 1234,
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
    }


def test_from_response_full() -> None:
    stats = CallStats.from_response(
        sample_id="s1",
        response=_full_response(),
        attempt_count=2,
    )
    assert stats.sample_id == "s1"
    assert stats.latency_ms == 1234
    assert stats.total_cost_usd == 0.005
    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 50
    assert stats.reasoning_tokens == 20
    assert stats.total_tokens == 150
    assert stats.attempt_count == 2
    assert stats.finish_reason == "stop"


def test_from_response_no_cost() -> None:
    response = _full_response()
    response["cost"] = None
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.total_cost_usd is None


def test_from_response_reasoning_tokens_zero_becomes_none() -> None:
    response = _full_response()
    response["usage"]["reasoning_tokens"] = 0
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.reasoning_tokens is None


def test_from_response_reasoning_tokens_nonzero_preserved() -> None:
    response = _full_response()
    response["usage"]["reasoning_tokens"] = 42
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.reasoning_tokens == 42


def test_from_response_missing_usage() -> None:
    response = {"text": "hi", "latency_ms": 500, "provider": "test", "model": "m"}
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.prompt_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0
    assert stats.reasoning_tokens is None
    assert stats.total_cost_usd is None
    assert stats.finish_reason is None
    assert stats.latency_ms == 500


@pytest.mark.parametrize("usage_value", [None, "bad"])
def test_from_response_malformed_usage(usage_value: object) -> None:
    response = {
        "text": "hi",
        "usage": usage_value,
        "latency_ms": 500,
        "provider": "test",
        "model": "m",
    }
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.prompt_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0
    assert stats.reasoning_tokens is None
    assert stats.total_cost_usd is None
    assert stats.finish_reason is None
    assert stats.latency_ms == 500


def test_from_response_missing_cost_key() -> None:
    response = _full_response()
    del response["cost"]
    stats = CallStats.from_response(
        sample_id="s1", response=response, attempt_count=1
    )
    assert stats.total_cost_usd is None
