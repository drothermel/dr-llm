from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from llm_pool.errors import ProviderSemanticError, ProviderTransportError
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.utils import parse_tool_calls, parse_usage, to_openai_messages
from llm_pool.types import CallMode, LlmRequest, LlmResponse


class OpenAICompatConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    chat_path: str = "/chat/completions"
    capabilities: ProviderCapabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_structured_output=True,
    )


class OpenAICompatAdapter(ProviderAdapter):
    mode = "api"

    def __init__(self, *, name: str, config: OpenAICompatConfig, client: httpx.Client | None = None) -> None:
        self.name = name
        self._config = config
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._config.capabilities

    def _headers(self) -> dict[str, str]:
        key = self._config.api_key or os.getenv(self._config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing API key for {self.name}. Set {self._config.api_key_env} or pass config.api_key"
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, ProviderTransportError)),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": to_openai_messages(request.messages),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        endpoint = self._config.base_url.rstrip("/") + self._config.chat_path
        started = time.perf_counter()
        try:
            resp = self._client.post(endpoint, headers=self._headers(), json=payload)
        except (httpx.TimeoutException, httpx.TransportError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderTransportError(f"{self.name} request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderTransportError(
                f"{self.name} transient error status={resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderSemanticError(
                f"{self.name} request rejected status={resp.status_code} body={resp.text[:500]}"
            )
        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderSemanticError(f"{self.name} response missing choices")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
        usage_raw = body.get("usage") or {}
        usage = parse_usage(
            prompt_tokens=usage_raw.get("prompt_tokens"),
            completion_tokens=usage_raw.get("completion_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )
        tool_calls = parse_tool_calls(message.get("tool_calls"))
        return LlmResponse(
            text=text,
            finish_reason=choices[0].get("finish_reason"),
            usage=usage,
            raw_json=body,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=tool_calls,
        )
