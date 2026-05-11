from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.transports.api_config import APIProviderConfig
from dr_llm.llm.providers.transports.api_provider import ApiProvider
from dr_llm.llm.providers.impls.google.request import GoogleRequest
from dr_llm.llm.providers.impls.google.response import GoogleResponse
from dr_llm.llm.request import LlmRequest


class GoogleUrls(StrEnum):
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    MODELS_DOCS = "https://ai.google.dev/gemini-api/docs/models"


class GoogleProvider(ApiProvider):
    def __init__(
        self,
        config: APIProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or APIProviderConfig(
                name=ProviderName.GOOGLE,
                base_url=GoogleUrls.API_BASE,
                api_key_env=ApiKeyNames.GOOGLE,
            ),
            client=client,
        )

    def _build_request(self, request: LlmRequest) -> GoogleRequest:
        return GoogleRequest.from_llm_request(request, self._config)

    def _parse_response(self, response: httpx.Response) -> GoogleResponse:
        return GoogleResponse.from_http_response(response)
