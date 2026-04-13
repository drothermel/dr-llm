"""Template-method base for HTTP API providers.

Provides shared client lifecycle, retry policy, and ``generate()`` flow. Concrete
subclasses supply only the request builder and response parser.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Protocol, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.request import ApiBackedLlmRequest, LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.logging.sinks import emit_generation_event


class ApiProviderRequest(Protocol):
    """Structural type for provider request objects sent over HTTP."""

    warnings: list[ReasoningWarning]

    def endpoint(self) -> str: ...

    def headers(self) -> dict[str, str]: ...

    def json_payload(self) -> dict[str, Any]: ...


class ApiProviderResponse(Protocol):
    """Structural type for provider response objects parsed from HTTP."""

    def event_payload(self, request: Any) -> dict[str, Any]: ...

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse: ...


class ApiProvider(Provider):
    """Template-method base shared by HTTP-backed providers."""

    _config: APIProviderConfig

    def __init__(
        self,
        *,
        config: APIProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def config(self) -> APIProviderConfig:
        return self._config

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @abstractmethod
    def _build_request(self, request: ApiBackedLlmRequest) -> ApiProviderRequest:
        """Translate an ``LlmRequest`` into the provider-specific request shape."""

    @abstractmethod
    def _parse_response(self, response: httpx.Response) -> ApiProviderResponse:
        """Decode an ``httpx.Response`` into the provider-specific response shape."""

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _post_with_retry(self, provider_request: ApiProviderRequest) -> httpx.Response:
        return self._client.post(
            provider_request.endpoint(),
            headers=provider_request.headers(),
            json=provider_request.json_payload(),
        )

    def generate(self, request: LlmRequest) -> LlmResponse:
        if not isinstance(request, ApiBackedLlmRequest):
            raise ProviderSemanticError(
                f"{self.name} only accepts API-backed request shapes"
            )
        provider_request = self._build_request(request)
        started = time.perf_counter()
        try:
            response = self._post_with_retry(provider_request)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderTransportError(
                f"{self.name} HTTP request failed: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        provider_response = self._parse_response(response)
        emit_generation_event(
            event_type="provider.raw_response",
            stage=f"{self.name}.http_response",
            payload=provider_response.event_payload(provider_request),
        )
        return provider_response.to_llm_response(
            request,
            latency_ms=latency_ms,
            warnings=provider_request.warnings,
        )
