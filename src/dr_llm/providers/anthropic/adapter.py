from __future__ import annotations

import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_llm.errors import ProviderTransportError
from dr_llm.generation.models import LlmRequest, LlmResponse
from dr_llm.logging import emit_generation_event
from dr_llm.providers.anthropic.config import AnthropicConfig
from dr_llm.providers.anthropic.request import AnthropicRequest
from dr_llm.providers.anthropic.response import AnthropicResponse
from dr_llm.providers.provider_adapter import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    _config: AnthropicConfig

    def __init__(
        self, config: AnthropicConfig | None = None, client: httpx.Client | None = None
    ) -> None:
        self._config = config or AnthropicConfig()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> AnthropicAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def config(self) -> AnthropicConfig:
        return self._config

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError, ProviderTransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        provider_request = AnthropicRequest.from_llm_request(request, self._config)
        started = time.perf_counter()
        response = self._client.post(
            provider_request.endpoint(),
            headers=provider_request.headers(),
            json=provider_request.json_payload(),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        provider_response = AnthropicResponse.from_http_response(response)
        emit_generation_event(
            event_type="provider.raw_response",
            stage="anthropic.http_response",
            payload=provider_response.event_payload(provider_request),
        )
        return provider_response.to_llm_response(
            request,
            latency_ms=latency_ms,
            warnings=provider_request.warnings,
        )
