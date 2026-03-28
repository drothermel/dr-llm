from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel

from dr_llm.types import CostInfo, Message

KEY_USAGE = "usage"
KEY_REASONING = "reasoning"
KEY_REASONING_CONTENT = "reasoning_content"
KEY_REASONING_DETAILS = "reasoning_details"
KEY_REASONING_TOKENS = "reasoning_tokens"
KEY_COMPLETION_TOKENS_DETAILS = "completion_tokens_details"
KEY_OUTPUT_TOKENS_DETAILS = "output_tokens_details"

KEY_COST = "cost"
KEY_TOTAL_COST = "total_cost"
KEY_PROMPT_COST = "prompt_cost"
KEY_COMPLETION_COST = "completion_cost"
KEY_REASONING_COST = "reasoning_cost"
KEY_CURRENCY = "currency"
DEFAULT_CURRENCY = "USD"
COST_INFO_KEYS = (
    KEY_COST,
    KEY_TOTAL_COST,
    KEY_PROMPT_COST,
    KEY_COMPLETION_COST,
    KEY_REASONING_COST,
    KEY_CURRENCY,
)
USAGE_PREFIX = "usage."
BODY_PREFIX = "body."


class _OpenAIWireMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    payloads = [
        _OpenAIWireMessage(role=message.role, content=message.content)
        for message in messages
    ]
    return [
        message.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        for message in payloads
    ]


def parse_reasoning_tokens(usage_raw: dict[str, Any] | None) -> int:
    if not isinstance(usage_raw, dict):
        return 0
    direct = _as_int(usage_raw.get(KEY_REASONING_TOKENS))
    if direct is not None:
        return direct

    completion_details = usage_raw.get(KEY_COMPLETION_TOKENS_DETAILS)
    if isinstance(completion_details, dict):
        value = _as_int(completion_details.get(KEY_REASONING_TOKENS))
        if value is not None:
            return value

    output_details = usage_raw.get(KEY_OUTPUT_TOKENS_DETAILS)
    if isinstance(output_details, dict):
        value = _as_int(output_details.get(KEY_REASONING_TOKENS))
        if value is not None:
            return value
    return 0


def parse_reasoning(
    message_raw: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    if not isinstance(message_raw, dict):
        return None, None

    reasoning_raw = message_raw.get(KEY_REASONING)
    if reasoning_raw is None:
        reasoning_raw = message_raw.get(KEY_REASONING_CONTENT)
    reasoning_text = (
        str(reasoning_raw) if isinstance(reasoning_raw, (str, int, float)) else None
    )

    reasoning_details_raw = message_raw.get(KEY_REASONING_DETAILS)
    reasoning_details: list[dict[str, Any]] | None = None
    if isinstance(reasoning_details_raw, list):
        reasoning_details = [
            item for item in reasoning_details_raw if isinstance(item, dict)
        ]
    return reasoning_text, reasoning_details


def parse_cost_info(body_raw: dict[str, Any] | None) -> CostInfo | None:
    if not isinstance(body_raw, dict):
        return None
    usage_raw = body_raw.get(KEY_USAGE)
    usage = usage_raw if isinstance(usage_raw, dict) else {}

    total_cost = _first_float(
        usage.get(KEY_TOTAL_COST),
        usage.get(KEY_COST),
        body_raw.get(KEY_TOTAL_COST),
        body_raw.get(KEY_COST),
    )
    prompt_cost = _first_float(
        usage.get(KEY_PROMPT_COST), body_raw.get(KEY_PROMPT_COST)
    )
    completion_cost = _first_float(
        usage.get(KEY_COMPLETION_COST), body_raw.get(KEY_COMPLETION_COST)
    )
    reasoning_cost = _first_float(
        usage.get(KEY_REASONING_COST), body_raw.get(KEY_REASONING_COST)
    )
    currency = _first_str(usage.get(KEY_CURRENCY), body_raw.get(KEY_CURRENCY))
    currency = currency or DEFAULT_CURRENCY

    if (
        total_cost is None
        and prompt_cost is None
        and completion_cost is None
        and reasoning_cost is None
    ):
        return None

    raw: dict[str, Any] = {}
    for key in COST_INFO_KEYS:
        if key in usage:
            raw[f"{USAGE_PREFIX}{key}"] = usage[key]
        if key in body_raw:
            raw[f"{BODY_PREFIX}{key}"] = body_raw[key]

    return CostInfo(
        total_cost_usd=total_cost,
        prompt_cost_usd=prompt_cost,
        completion_cost_usd=completion_cost,
        reasoning_cost_usd=reasoning_cost,
        currency=currency,
        raw=raw,
    )


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
