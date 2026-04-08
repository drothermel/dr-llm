from __future__ import annotations

import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.errors import ProviderTransportError
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.logging import emit_generation_event
from dr_llm.llm.providers.google.request import GoogleRequest
from dr_llm.llm.providers.google.response import GoogleResponse
from dr_llm.llm.providers.base import Provider


DEFAULT_GOOGLE_CONFIG = APIProviderConfig(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GOOGLE_API_KEY",
)


class GoogleProvider(Provider):
    _config: APIProviderConfig

    def __init__(
        self,
        config: APIProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or DEFAULT_GOOGLE_CONFIG
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def config(self) -> APIProviderConfig:
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
        provider_request = GoogleRequest.from_llm_request(request, self._config)
        started = time.perf_counter()
        resp = self._client.post(
            provider_request.endpoint(),
            headers=provider_request.headers(),
            json=provider_request.json_payload(),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        provider_response = GoogleResponse.from_http_response(resp)
        emit_generation_event(
            event_type="provider.raw_response",
            stage="google.http_response",
            payload=provider_response.event_payload(provider_request),
        )
        return provider_response.to_llm_response(
            request,
            latency_ms=latency_ms,
            warnings=provider_request.warnings,
        )
