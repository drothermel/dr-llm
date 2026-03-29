from __future__ import annotations

import json
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

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.generation.models import (
    CallMode,
    LlmRequest,
    LlmResponse,
    TokenUsage,
)
from dr_llm.logging import emit_generation_event
from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.openai_compat_request import OpenAICompatRequest
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)


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


class _OpenAICompatChoice(BaseModel):
    finish_reason: str | None = None
    message: _OpenAICompatMessage


class _OpenAICompatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None


class OpenAICompatConfig(APIProviderConfig):
    api_key_env: str = "OPENAI_API_KEY"
    chat_path: str = "/chat/completions"


class OpenAICompatAdapter(ProviderAdapter):
    _config: OpenAICompatConfig

    def __init__(
        self,
        *,
        config: OpenAICompatConfig,
        client: httpx.Client | None = None,
    ) -> None:
        """Create adapter with optional injected client.

        If no client is injected, the adapter owns the internally created client and
        closes it on replacement/close. Injected clients are treated as externally
        owned and are not closed by this adapter.
        """
        self._config = config
        self._client: httpx.Client | None = None
        self._owns_client = False
        self._set_client(
            client or httpx.Client(timeout=self._config.timeout_seconds),
            owns_client=client is None,
        )

    @property
    def config(self) -> OpenAICompatConfig:
        return self._config

    def _set_client(self, client: httpx.Client, *, owns_client: bool) -> None:
        if (
            self._client is not None
            and self._client is not client
            and self._owns_client
        ):
            self._client.close()
        self._client = client
        self._owns_client = owns_client

    def set_client(self, client: httpx.Client) -> None:
        self._set_client(client, owns_client=False)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None
        self._owns_client = False

    def __enter__(self) -> OpenAICompatAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

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
        provider_request = OpenAICompatRequest.from_llm_request(request, self._config)
        started = time.perf_counter()
        try:
            resp = self._client.post(
                provider_request.endpoint(),
                headers=provider_request.headers(),
                json=provider_request.json_payload(),
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderTransportError(
                f"{self.name} transport failure: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderTransportError(f"{self.name} request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        emit_generation_event(
            event_type="provider.raw_response",
            stage=f"{self.name}.http_response",
            payload={
                "status_code": resp.status_code,
                "endpoint": provider_request.endpoint(),
                "idempotency_key": provider_request.idempotency_key,
                "response_text_preview": resp.text[:500],
                "request_shape": {
                    "model": provider_request.model,
                    "message_count": len(provider_request.messages),
                },
            },
        )
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
            warnings=provider_request.warnings,
        )
