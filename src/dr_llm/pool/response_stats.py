"""Parse structured stats from pool sample response blobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_llm.pool.pool_sample import PoolSample


class LlmResponseStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    latency_ms: int | None = None
    total_cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    attempt_count: int | None = None


def parse_response_stats(
    response_json: dict[str, Any] | None,
    *,
    finish_reason: str | None = None,
    attempt_count: int = 0,
) -> LlmResponseStats:
    if response_json is None:
        return LlmResponseStats(
            finish_reason=finish_reason,
            attempt_count=attempt_count,
        )

    raw_usage = response_json.get("usage")
    raw_cost = response_json.get("cost")
    usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    cost = raw_cost if isinstance(raw_cost, Mapping) else {}

    return LlmResponseStats(
        latency_ms=response_json.get("latency_ms"),
        total_cost_usd=cost.get("total_cost_usd"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
        finish_reason=finish_reason,
        attempt_count=attempt_count,
    )


def sample_response_stats(sample: PoolSample) -> LlmResponseStats:
    return parse_response_stats(
        sample.response,
        finish_reason=sample.finish_reason,
        attempt_count=sample.attempt_count,
    )
