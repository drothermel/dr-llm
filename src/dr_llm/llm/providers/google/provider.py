from __future__ import annotations

import httpx

from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.api_provider import ApiProvider
from dr_llm.llm.providers.google.request import GoogleRequest
from dr_llm.llm.providers.google.response import GoogleResponse
from dr_llm.llm.request import ApiLlmRequest


DEFAULT_GOOGLE_CONFIG = APIProviderConfig(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GOOGLE_API_KEY",
)


class GoogleProvider(ApiProvider):
    def __init__(
        self,
        config: APIProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(config=config or DEFAULT_GOOGLE_CONFIG, client=client)

    def _build_request(self, request: ApiLlmRequest) -> GoogleRequest:
        return GoogleRequest.from_llm_request(request, self._config)

    def _parse_response(self, response: httpx.Response) -> GoogleResponse:
        return GoogleResponse.from_http_response(response)
