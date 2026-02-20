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


class GoogleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GOOGLE_API_KEY"
    api_key: str | None = None
    timeout_seconds: float = 120.0


class GoogleAdapter(ProviderAdapter):
    name = "google"
    mode = "api"

    def __init__(self, config: GoogleConfig | None = None, client: httpx.Client | None = None) -> None:
        self._config = config or GoogleConfig()
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_native_tools=True, supports_structured_output=True)

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, ProviderTransportError)),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        key = self._config.api_key or os.getenv(self._config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing Google API key. Set {self._config.api_key_env} or pass config.api_key"
            )
        endpoint = (
            f"{self._config.base_url}/models/{request.model}:generateContent"
            f"?key={key}"
        )
        system = "\n".join(msg.content for msg in request.messages if msg.role == "system")
        contents = [
            {
                "role": "model" if msg.role == "assistant" else "user",
                "parts": [{"text": msg.content}],
            }
            for msg in request.messages
            if msg.role in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if request.temperature is not None or request.top_p is not None or request.max_tokens is not None:
            payload["generationConfig"] = {}
            if request.temperature is not None:
                payload["generationConfig"]["temperature"] = request.temperature
            if request.top_p is not None:
                payload["generationConfig"]["topP"] = request.top_p
            if request.max_tokens is not None:
                payload["generationConfig"]["maxOutputTokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools

        started = time.perf_counter()
        resp = self._client.post(endpoint, json=payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderTransportError(
                f"google transient error status={resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderSemanticError(
                f"google rejected request status={resp.status_code} body={resp.text[:500]}"
            )
        body = resp.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ProviderSemanticError("google response missing candidates")
        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_chunks: list[str] = []
        tool_calls: list[ModelToolCall] = []
        for idx, part in enumerate(parts):
            if "text" in part:
                text_chunks.append(str(part.get("text") or ""))
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                tool_calls.append(
                    ModelToolCall(
                        tool_call_id=f"google_call_{idx+1}",
                        name=str(fc.get("name") or ""),
                        arguments=fc.get("args") if isinstance(fc.get("args"), dict) else {},
                    )
                )
        usage_raw = body.get("usageMetadata") or {}
        usage = parse_usage(
            prompt_tokens=usage_raw.get("promptTokenCount"),
            completion_tokens=usage_raw.get("candidatesTokenCount"),
            total_tokens=usage_raw.get("totalTokenCount"),
        )
        return LlmResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk),
            finish_reason=candidate.get("finishReason"),
            usage=usage,
            raw_json=body,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=[tc for tc in tool_calls if tc.name],
        )
