from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_pool.errors import ProviderSemanticError, ProviderTransportError
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
    parse_tool_calls,
    to_openai_messages,
)
from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ProviderToolSpec,
    TokenUsage,
)


class _OpenAICompatRequestPayload(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    tools: list[ProviderToolSpec] | None = None


class _OpenAICompatUsageDetails(BaseModel):
    reasoning_tokens: int | None = None


class _OpenAICompatUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: _OpenAICompatUsageDetails | None = None
    output_tokens_details: _OpenAICompatUsageDetails | None = None


class _OpenAICompatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str | None = None
    reasoning: str | int | float | None = None
    reasoning_content: str | int | float | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _OpenAICompatChoice(BaseModel):
    finish_reason: str | None = None
    message: _OpenAICompatMessage


class _OpenAICompatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None


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

    def __init__(
        self,
        *,
        name: str,
        config: OpenAICompatConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self._config = config
        self._client: httpx.Client | None = None
        self._set_client(client or httpx.Client(timeout=self._config.timeout_seconds))

    def _set_client(self, client: httpx.Client) -> None:
        if self._client is not None and self._client is not client:
            self._client.close()
        self._client = client

    def set_client(self, client: httpx.Client) -> None:
        self._set_client(client)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> OpenAICompatAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

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
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError, ProviderTransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        if self._client is None:
            raise ProviderTransportError(f"{self.name} client is not initialized")
        payload = _OpenAICompatRequestPayload(
            model=request.model,
            messages=to_openai_messages(request.messages),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            reasoning=(
                request.reasoning.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_computed_fields=True,
                )
                if request.reasoning is not None
                else None
            ),
            tools=request.tools,
        ).model_dump(mode="json", exclude_none=True)
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
        try:
            body_raw = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderTransportError(
                f"{self.name} invalid JSON response: {exc}"
            ) from exc
        if not isinstance(body_raw, dict):
            raise ProviderSemanticError(
                f"{self.name} response shape invalid: expected JSON object"
            )
        try:
            body = _OpenAICompatResponse(**body_raw)
        except ValidationError as exc:
            raise ProviderSemanticError(
                f"{self.name} response shape invalid: {exc}"
            ) from exc
        if not body.choices:
            raise ProviderSemanticError(f"{self.name} response missing choices")
        choice = body.choices[0]
        message = choice.message
        text = message.content or ""
        usage_raw = (
            body.usage.model_dump(mode="json", exclude_none=True) if body.usage else {}
        )
        reasoning_tokens = parse_reasoning_tokens(
            usage_raw if isinstance(usage_raw, dict) else {}
        )
        usage = TokenUsage.from_raw(
            prompt_tokens=body.usage.prompt_tokens if body.usage else None,
            completion_tokens=body.usage.completion_tokens if body.usage else None,
            total_tokens=body.usage.total_tokens if body.usage else None,
            reasoning_tokens=reasoning_tokens,
        )
        reasoning, reasoning_details = parse_reasoning(
            message.model_dump(mode="json", exclude_none=True)
        )
        cost = parse_cost_info(body_raw if isinstance(body_raw, dict) else {})
        tool_calls = parse_tool_calls(message.tool_calls)
        return LlmResponse(
            text=text,
            finish_reason=choice.finish_reason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=cost,
            raw_json=body_raw if isinstance(body_raw, dict) else {"body": body_raw},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=tool_calls,
        )
