from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class CallStats(BaseModel):
    """Per-call metrics extracted at promotion time."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    latency_ms: int
    total_cost_usd: float | None = None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None = None
    total_tokens: int
    attempt_count: int = 1
    finish_reason: str | None = None

    @classmethod
    def from_response(
        cls,
        *,
        sample_id: str,
        response: dict[str, Any],
        attempt_count: int,
    ) -> CallStats:
        """Extract call stats from a ``response.model_dump()`` dict."""
        raw_usage = response.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        raw_cost = response.get("cost")
        cost = raw_cost if isinstance(raw_cost, dict) else {}
        raw_reasoning = usage.get("reasoning_tokens", 0)
        return cls(
            sample_id=sample_id,
            latency_ms=response.get("latency_ms", 0),
            total_cost_usd=cost.get("total_cost_usd"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=raw_reasoning if raw_reasoning else None,
            total_tokens=usage.get("total_tokens", 0),
            attempt_count=attempt_count,
            finish_reason=response.get("finish_reason"),
        )
