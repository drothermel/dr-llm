from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from llm_pool.types import CostInfo, Message, ModelToolCall, TokenUsage


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=True, sort_keys=True
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        payloads.append(item)
    return payloads


def parse_usage(
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> TokenUsage:
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    t = int(total_tokens or (p + c))
    r = int(reasoning_tokens or 0)
    return TokenUsage(
        prompt_tokens=p, completion_tokens=c, total_tokens=t, reasoning_tokens=r
    )


def parse_reasoning_tokens(usage_raw: dict[str, Any] | None) -> int:
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


def parse_reasoning(
    message_raw: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
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
    return reasoning_text, reasoning_details


def parse_cost_info(body_raw: dict[str, Any] | None) -> CostInfo | None:
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
    prompt_cost = _first_float(usage.get("prompt_cost"), body_raw.get("prompt_cost"))
    completion_cost = _first_float(
        usage.get("completion_cost"), body_raw.get("completion_cost")
    )
    reasoning_cost = _first_float(
        usage.get("reasoning_cost"), body_raw.get("reasoning_cost")
    )
    currency = _first_str(usage.get("currency"), body_raw.get("currency")) or "USD"

    if (
        total_cost is None
        and prompt_cost is None
        and completion_cost is None
        and reasoning_cost is None
    ):
        return None

    raw: dict[str, Any] = {}
    for key in (
        "cost",
        "total_cost",
        "prompt_cost",
        "completion_cost",
        "reasoning_cost",
        "currency",
    ):
        if key in usage:
            raw[f"usage.{key}"] = usage[key]
        if key in body_raw:
            raw[f"body.{key}"] = body_raw[key]

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
        call_id = str(item.get("id") or item.get("tool_call_id") or "")
        fn = item.get("function") or {}
        name = str(fn.get("name") or item.get("name") or "")
        args_raw = (
            fn.get("arguments") if isinstance(fn, dict) else item.get("arguments")
        )
        args: dict[str, Any]
        if isinstance(args_raw, str):
            try:
                loaded = json.loads(args_raw)
            except json.JSONDecodeError:
                loaded = {"_raw": args_raw}
            args = loaded if isinstance(loaded, dict) else {"_value": loaded}
        elif isinstance(args_raw, dict):
            args = args_raw
        elif args_raw is None:
            args = {}
        else:
            args = {"_value": args_raw}
        if not name:
            continue
        parsed.append(
            ModelToolCall(
                tool_call_id=call_id or f"call_{len(parsed) + 1}",
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
