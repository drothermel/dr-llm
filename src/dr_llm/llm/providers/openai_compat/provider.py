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
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.logging import emit_generation_event
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat.request import OpenAICompatRequest
from dr_llm.llm.providers.openai_compat.response import OpenAICompatResponse
from dr_llm.llm.providers.base import Provider


class OpenAICompatProvider(Provider):
    _config: OpenAICompatConfig

    def __init__(
        self,
        *,
        config: OpenAICompatConfig,
        client: httpx.Client | None = None,
    ) -> None:
        """Create provider with optional injected client.

        If no client is injected, the provider owns the internally created client and
        closes it on replacement/close. Injected clients are treated as externally
        owned and are not closed by this provider.
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

    def __enter__(self) -> OpenAICompatProvider:
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
        provider_response = OpenAICompatResponse.from_http_response(resp)
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
