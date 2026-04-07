from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _first_float(*values: object) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_str(*values: object) -> str | None:
    for value in values:
        parsed = _as_str(value)
        if parsed is not None:
            return parsed
    return None


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)

    @classmethod
    def _coerce_token_count(cls, value: Any, *, field_name: str) -> int:
        if value is None:
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if parsed < 0:
            raise ValueError("token counts must be non-negative")
        return parsed

    @model_validator(mode="before")
    @classmethod
    def _normalize_counts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prompt_tokens = cls._coerce_token_count(
            data.get("prompt_tokens"), field_name="prompt_tokens"
        )
        completion_tokens = cls._coerce_token_count(
            data.get("completion_tokens"), field_name="completion_tokens"
        )
        reasoning_tokens = cls._coerce_token_count(
            data.get("reasoning_tokens"), field_name="reasoning_tokens"
        )
        total_raw = data.get("total_tokens")
        total_tokens = (
            prompt_tokens + completion_tokens
            if total_raw is None
            else cls._coerce_token_count(total_raw, field_name="total_tokens")
        )
        return {
            **data,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    @classmethod
    def from_raw(
        cls,
        *,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        total_tokens: Any = None,
        reasoning_tokens: Any = None,
    ) -> TokenUsage:
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @computed_field
    @property
    def computed_total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def extract_reasoning_tokens(cls, usage_raw: dict[str, Any] | None) -> int:
        """Extract reasoning token count from a raw usage dict.

        Checks three locations in priority order:
        1. usage.reasoning_tokens
        2. usage.completion_tokens_details.reasoning_tokens
        3. usage.output_tokens_details.reasoning_tokens
        """
        if not isinstance(usage_raw, dict):
            return 0
        direct = _as_int(usage_raw.get("reasoning_tokens"))
        if direct is not None:
            return direct

        completion_details = usage_raw.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            value = _as_int(completion_details.get("reasoning_tokens"))
            if value is not None:
                return value

        output_details = usage_raw.get("output_tokens_details")
        if isinstance(output_details, dict):
            value = _as_int(output_details.get("reasoning_tokens"))
            if value is not None:
                return value
        return 0


# ---------------------------------------------------------------------------
# CostInfo
# ---------------------------------------------------------------------------

_DEFAULT_CURRENCY = "USD"
_COST_KEYS = (
    "cost",
    "total_cost",
    "prompt_cost",
    "completion_cost",
    "reasoning_cost",
    "currency",
)


class CostInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cost_usd: float | None = None
    prompt_cost_usd: float | None = None
    completion_cost_usd: float | None = None
    reasoning_cost_usd: float | None = None
    currency: str | None = "USD"
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, body_raw: dict[str, Any] | None) -> CostInfo | None:
        """Parse cost information from a raw response body dict."""
        if not isinstance(body_raw, dict):
            return None
        usage_raw = body_raw.get("usage")
        usage = usage_raw if isinstance(usage_raw, dict) else {}

        total_cost = _first_float(
            usage.get("total_cost"),
            usage.get("cost"),
            body_raw.get("total_cost"),
            body_raw.get("cost"),
        )
        prompt_cost = _first_float(
            usage.get("prompt_cost"), body_raw.get("prompt_cost")
        )
        completion_cost = _first_float(
            usage.get("completion_cost"), body_raw.get("completion_cost")
        )
        reasoning_cost = _first_float(
            usage.get("reasoning_cost"), body_raw.get("reasoning_cost")
        )
        currency = _first_str(usage.get("currency"), body_raw.get("currency"))
        currency = currency or _DEFAULT_CURRENCY

        if (
            total_cost is None
            and prompt_cost is None
            and completion_cost is None
            and reasoning_cost is None
        ):
            return None

        raw: dict[str, Any] = {}
        for key in _COST_KEYS:
            if key in usage:
                raw[f"usage.{key}"] = usage[key]
            if key in body_raw:
                raw[f"body.{key}"] = body_raw[key]

        is_usd = currency == _DEFAULT_CURRENCY
        return cls(
            total_cost_usd=total_cost if is_usd else None,
            prompt_cost_usd=prompt_cost if is_usd else None,
            completion_cost_usd=completion_cost if is_usd else None,
            reasoning_cost_usd=reasoning_cost if is_usd else None,
            currency=currency,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Reasoning text/details parsing
# ---------------------------------------------------------------------------


def parse_reasoning(
    message_raw: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Extract reasoning text and details from a raw message dict."""
    if not isinstance(message_raw, dict):
        return None, None

    reasoning_raw = message_raw.get("reasoning")
    if reasoning_raw is None:
        reasoning_raw = message_raw.get("reasoning_content")
    reasoning_text = (
        str(reasoning_raw) if isinstance(reasoning_raw, (str, int, float)) else None
    )

    reasoning_details_raw = message_raw.get("reasoning_details")
    reasoning_details: list[dict[str, Any]] | None = None
    if isinstance(reasoning_details_raw, list):
        reasoning_details = [
            item for item in reasoning_details_raw if isinstance(item, dict)
        ]
    if reasoning_text is None:
        content_raw = message_raw.get("content")
        if isinstance(content_raw, list):
            thinking_items = [
                item
                for item in content_raw
                if isinstance(item, dict) and item.get("type") == "thinking"
            ]
            if thinking_items:
                reasoning_chunks = [
                    val
                    for item in thinking_items
                    if (val := _as_str(item.get("thinking"))) is not None
                ]
                if reasoning_chunks:
                    reasoning_text = "\n\n".join(reasoning_chunks)
    return reasoning_text, reasoning_details
