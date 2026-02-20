from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from llm_pool.types import Message, ModelToolCall, TokenUsage


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": message.content,
            **({"name": message.name} if message.name else {}),
        }
        for message in messages
    ]


def parse_usage(
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> TokenUsage:
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    t = int(total_tokens or (p + c))
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=t)


def parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ModelToolCall]:
    if not raw:
        return []
    parsed: list[ModelToolCall] = []
    for item in raw:
        call_id = str(item.get("id") or item.get("tool_call_id") or "")
        fn = item.get("function") or {}
        name = str(fn.get("name") or item.get("name") or "")
        args_raw = fn.get("arguments") if isinstance(fn, dict) else item.get("arguments")
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
                tool_call_id=call_id or f"call_{len(parsed)+1}",
                name=name,
                arguments=args,
            )
        )
    return parsed
