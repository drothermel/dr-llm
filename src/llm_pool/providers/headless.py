from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_pool.errors import HeadlessExecutionError
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
    parse_usage,
)
from llm_pool.types import CallMode, LlmRequest, LlmResponse, ModelToolCall


class HeadlessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: list[str]
    timeout_seconds: float = 180.0
    env_overrides: dict[str, str] = Field(default_factory=dict)


class _BaseHeadlessAdapter(ProviderAdapter):
    mode = "headless"

    def __init__(self, *, name: str, config: HeadlessConfig) -> None:
        self.name = name
        self._config = config

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_native_tools=False, supports_structured_output=True
        )

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        return {
            "provider": request.provider,
            "model": request.model,
            "messages": [msg.model_dump(mode="json") for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "reasoning": request.reasoning.model_dump(mode="json", exclude_none=True)
            if request.reasoning
            else None,
            "metadata": request.metadata,
            "tools": request.tools,
            "tool_policy": request.tool_policy.value,
        }

    def generate(self, request: LlmRequest) -> LlmResponse:
        payload = self._payload(request)
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                self._config.command,
                input=json.dumps(payload, ensure_ascii=True),
                text=True,
                capture_output=True,
                timeout=self._config.timeout_seconds,
                env={**os.environ, **self._config.env_overrides},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HeadlessExecutionError(
                f"{self.name} command timed out after {self._config.timeout_seconds}s"
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if proc.returncode != 0:
            raise HeadlessExecutionError(
                f"{self.name} command failed with exit code {proc.returncode}: {proc.stderr[:800]}"
            )

        stdout = proc.stdout.strip()
        if not stdout:
            return LlmResponse(
                text="",
                finish_reason=None,
                usage=parse_usage(),
                raw_json={"stdout": "", "stderr": proc.stderr},
                latency_ms=latency_ms,
                provider=request.provider,
                model=request.model,
                mode=CallMode.headless,
            )

        try:
            body = json.loads(stdout)
        except json.JSONDecodeError:
            return LlmResponse(
                text=stdout,
                finish_reason=None,
                usage=parse_usage(),
                raw_json={"stdout": stdout, "stderr": proc.stderr},
                latency_ms=latency_ms,
                provider=request.provider,
                model=request.model,
                mode=CallMode.headless,
            )

        raw_usage = body.get("usage") if isinstance(body, dict) else None
        reasoning_tokens = parse_reasoning_tokens(
            raw_usage if isinstance(raw_usage, dict) else {}
        )
        usage = parse_usage(
            prompt_tokens=(raw_usage or {}).get("prompt_tokens")
            if isinstance(raw_usage, dict)
            else None,
            completion_tokens=(raw_usage or {}).get("completion_tokens")
            if isinstance(raw_usage, dict)
            else None,
            total_tokens=(raw_usage or {}).get("total_tokens")
            if isinstance(raw_usage, dict)
            else None,
            reasoning_tokens=reasoning_tokens,
        )
        message_raw = (
            body.get("message")
            if isinstance(body, dict) and isinstance(body.get("message"), dict)
            else body
        )
        reasoning, reasoning_details = parse_reasoning(
            message_raw if isinstance(message_raw, dict) else None
        )
        if (
            reasoning is None
            and isinstance(body, dict)
            and isinstance(body.get("reasoning"), str)
        ):
            reasoning = body.get("reasoning")
        if (
            reasoning_details is None
            and isinstance(body, dict)
            and isinstance(body.get("reasoning_details"), list)
        ):
            reasoning_details = [
                item for item in body.get("reasoning_details") if isinstance(item, dict)
            ]
        cost = parse_cost_info(body if isinstance(body, dict) else {})
        tool_calls: list[ModelToolCall] = []
        for idx, item in enumerate(
            (body.get("tool_calls") or []) if isinstance(body, dict) else []
        ):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not name:
                continue
            args = (
                item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            )
            tool_calls.append(
                ModelToolCall(
                    tool_call_id=str(
                        item.get("tool_call_id") or item.get("id") or f"call_{idx + 1}"
                    ),
                    name=name,
                    arguments=args,
                )
            )

        return LlmResponse(
            text=str(body.get("text") or ""),
            finish_reason=body.get("finish_reason"),
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=cost,
            raw_json=body if isinstance(body, dict) else {"body": body},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.headless,
            tool_calls=tool_calls,
        )


class CodexHeadlessAdapter(_BaseHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            name="codex",
            config=HeadlessConfig(command=command or ["codex", "--headless", "--json"]),
        )


class ClaudeHeadlessAdapter(_BaseHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            name="claude-code",
            config=HeadlessConfig(
                command=command or ["claude", "--print", "--output-format", "json"]
            ),
        )
