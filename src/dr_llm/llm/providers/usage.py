from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from dr_llm.llm.coercion import as_int, as_str, first_float, first_str


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
        direct = as_int(usage_raw.get("reasoning_tokens"))
        if direct is not None:
            return direct

        completion_details = usage_raw.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            value = as_int(completion_details.get("reasoning_tokens"))
            if value is not None:
                return value

        output_details = usage_raw.get("output_tokens_details")
        if isinstance(output_details, dict):
            value = as_int(output_details.get("reasoning_tokens"))
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

        total_cost = _find_float_value(usage, body_raw, "total_cost", "cost")
        prompt_cost = _find_float_value(usage, body_raw, "prompt_cost")
        completion_cost = _find_float_value(usage, body_raw, "completion_cost")
        reasoning_cost = _find_float_value(usage, body_raw, "reasoning_cost")
        currency = _find_string_value(usage, body_raw, "currency") or _DEFAULT_CURRENCY

        if (
            total_cost is None
            and prompt_cost is None
            and completion_cost is None
            and reasoning_cost is None
        ):
            return None

        is_usd = currency == _DEFAULT_CURRENCY
        return cls(
            total_cost_usd=total_cost if is_usd else None,
            prompt_cost_usd=prompt_cost if is_usd else None,
            completion_cost_usd=completion_cost if is_usd else None,
            reasoning_cost_usd=reasoning_cost if is_usd else None,
            currency=currency,
            raw=_collect_cost_provenance(usage, body_raw),
        )


def _find_float_value(
    usage: dict[str, Any],
    body_raw: dict[str, Any],
    *keys: str,
) -> float | None:
    """Look up the first float-coercible value across `usage` then `body_raw` for each key."""
    candidates = [usage.get(key) for key in keys] + [body_raw.get(key) for key in keys]
    return first_float(*candidates)


def _find_string_value(
    usage: dict[str, Any],
    body_raw: dict[str, Any],
    *keys: str,
) -> str | None:
    """Look up the first string-coercible value across `usage` then `body_raw` for each key."""
    candidates = [usage.get(key) for key in keys] + [body_raw.get(key) for key in keys]
    return first_str(*candidates)


def _collect_cost_provenance(
    usage: dict[str, Any],
    body_raw: dict[str, Any],
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key in _COST_KEYS:
        if key in usage:
            raw[f"usage.{key}"] = usage[key]
        if key in body_raw:
            raw[f"body.{key}"] = body_raw[key]
    return raw


# ---------------------------------------------------------------------------
# Reasoning text/details parsing
# ---------------------------------------------------------------------------


def build_usage_and_reasoning(
    *,
    usage_dump: dict[str, Any] | None,
    prompt_tokens: Any,
    completion_tokens: Any,
    total_tokens: Any,
    reasoning_source: dict[str, Any] | None,
) -> tuple[TokenUsage, str | None, list[dict[str, Any]] | None]:
    """Build a ``TokenUsage`` plus parsed reasoning text/details from raw fields.

    ``usage_dump`` is the raw usage dict (used to extract reasoning token counts
    from nested ``*_tokens_details`` fields). The explicit token args supply the
    final prompt/completion/total counts. ``reasoning_source`` is the dict to
    scan for ``reasoning`` / ``reasoning_details`` (typically the message body or
    raw response body).
    """
    reasoning_tokens = TokenUsage.extract_reasoning_tokens(usage_dump or {})
    usage = TokenUsage.from_raw(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    reasoning, reasoning_details = parse_reasoning(reasoning_source)
    return usage, reasoning, reasoning_details


def parse_reasoning(
    message_raw: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Extract reasoning text and details from a raw message dict."""
    if not isinstance(message_raw, dict):
        return None, None
    reasoning_text, reasoning_details = _extract_direct_reasoning(message_raw)
    if reasoning_text is None:
        reasoning_text = _extract_thinking_from_content(message_raw)
    return reasoning_text, reasoning_details


def _extract_direct_reasoning(
    message_raw: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None]:
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
    return reasoning_text, reasoning_details


def _extract_thinking_from_content(message_raw: dict[str, Any]) -> str | None:
    content_raw = message_raw.get("content")
    if not isinstance(content_raw, list):
        return None
    thinking_chunks = [
        val
        for item in content_raw
        if isinstance(item, dict) and item.get("type") == "thinking"
        if (val := as_str(item.get("thinking"))) is not None
    ]
    if not thinking_chunks:
        return None
    return "\n\n".join(thinking_chunks)
