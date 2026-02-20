from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from llm_pool.errors import ProviderSemanticError, ProviderTransportError
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.utils import parse_usage
from llm_pool.types import CallMode, LlmRequest, LlmResponse, ModelToolCall


class AnthropicConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "https://api.anthropic.com/v1/messages"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: str | None = None
    anthropic_version: str = "2023-06-01"
    timeout_seconds: float = 120.0


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    mode = "api"

    def __init__(self, config: AnthropicConfig | None = None, client: httpx.Client | None = None) -> None:
        self._config = config or AnthropicConfig()
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_native_tools=True, supports_structured_output=True)

    def _headers(self) -> dict[str, str]:
        key = self._config.api_key or os.getenv(self._config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing Anthropic API key. Set {self._config.api_key_env} or pass config.api_key"
            )
        return {
            "x-api-key": key,
            "anthropic-version": self._config.anthropic_version,
            "content-type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, ProviderTransportError)),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        system = "\n".join(msg.content for msg in request.messages if msg.role == "system")
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
            if msg.role in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.tools:
            payload["tools"] = request.tools
        started = time.perf_counter()
        resp = self._client.post(self._config.base_url, headers=self._headers(), json=payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderTransportError(
                f"anthropic transient error status={resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderSemanticError(
                f"anthropic rejected request status={resp.status_code} body={resp.text[:500]}"
            )

        body = resp.json()
        content = body.get("content") or []
        text_chunks: list[str] = []
        tool_calls: list[ModelToolCall] = []
        for idx, item in enumerate(content):
            item_type = item.get("type")
            if item_type == "text":
                text_chunks.append(str(item.get("text") or ""))
            elif item_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        tool_call_id=str(item.get("id") or f"call_{idx+1}"),
                        name=str(item.get("name") or ""),
                        arguments=item.get("input") if isinstance(item.get("input"), dict) else {},
                    )
                )

        usage_raw = body.get("usage") or {}
        usage = parse_usage(
            prompt_tokens=usage_raw.get("input_tokens"),
            completion_tokens=usage_raw.get("output_tokens"),
            total_tokens=(usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0)),
        )
        return LlmResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk),
            finish_reason=body.get("stop_reason"),
            usage=usage,
            raw_json=body,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=[tc for tc in tool_calls if tc.name],
        )
