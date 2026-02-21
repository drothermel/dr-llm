from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel
from llm_pool.types import CostInfo, Message, ModelToolCall

KEY_NAME = "name"
KEY_TOOL_CALL_ID = "tool_call_id"
KEY_ID = "id"
KEY_FUNCTION = "function"
KEY_ARGUMENTS = "arguments"

TOOL_TYPE_FUNCTION = "function"
TOOL_CALL_ID_PREFIX = "call_"
FALLBACK_RAW_ARGUMENT_KEY = "_raw"
FALLBACK_VALUE_ARGUMENT_KEY = "_value"

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


class _OpenAIWireToolFunction(BaseModel):
    name: str
    arguments: str


class _OpenAIWireToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: _OpenAIWireToolFunction


class _OpenAIWireMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[_OpenAIWireToolCall] | None = None


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    payloads: list[_OpenAIWireMessage] = []
    for message in messages:
        payloads.append(
            _OpenAIWireMessage(
                role=message.role,
                content=message.content,
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=[
                    _OpenAIWireToolCall(
                        id=call.tool_call_id,
                        type=TOOL_TYPE_FUNCTION,
                        function=_OpenAIWireToolFunction(
                            name=call.name,
                            arguments=json.dumps(
                                call.arguments, ensure_ascii=True, sort_keys=True
                            ),
                        ),
                    )
                    for call in (message.tool_calls or [])
                ]
                or None,
            )
        )
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


def parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ModelToolCall]:
    if not raw:
        return []
    parsed: list[ModelToolCall] = []
    for item in raw:
        call_id = str(item.get(KEY_ID) or item.get(KEY_TOOL_CALL_ID) or "")
        fn = item.get(KEY_FUNCTION) or {}
        name = str(fn.get(KEY_NAME) or item.get(KEY_NAME) or "")
        args_raw = (
            fn.get(KEY_ARGUMENTS) if isinstance(fn, dict) else item.get(KEY_ARGUMENTS)
        )
        args: dict[str, Any]
        if isinstance(args_raw, str):
            try:
                loaded = json.loads(args_raw)
            except json.JSONDecodeError:
                loaded = {FALLBACK_RAW_ARGUMENT_KEY: args_raw}
            args = (
                loaded
                if isinstance(loaded, dict)
                else {FALLBACK_VALUE_ARGUMENT_KEY: loaded}
            )
        elif isinstance(args_raw, dict):
            args = args_raw
        elif args_raw is None:
            args = {}
        else:
            args = {FALLBACK_VALUE_ARGUMENT_KEY: args_raw}
        if not name:
            continue
        parsed.append(
            ModelToolCall(
                tool_call_id=call_id or f"{TOOL_CALL_ID_PREFIX}{len(parsed) + 1}",
                name=name,
                arguments=args,
            )
        )
    return parsed


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


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None
