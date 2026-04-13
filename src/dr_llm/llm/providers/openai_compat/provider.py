from __future__ import annotations

import httpx

from dr_llm.llm.providers.api_provider import ApiProvider
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat.request import OpenAICompatRequest
from dr_llm.llm.providers.openai_compat.response import OpenAICompatResponse
from dr_llm.llm.request import ApiBackedLlmRequest


class OpenAICompatProvider(ApiProvider):
    _config: OpenAICompatConfig

    def __init__(
        self,
        *,
        config: OpenAICompatConfig,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(config=config, client=client)

    @property
    def config(self) -> OpenAICompatConfig:
        return self._config

    def _build_request(self, request: ApiBackedLlmRequest) -> OpenAICompatRequest:
        return OpenAICompatRequest.from_llm_request(request, self._config)

    def _parse_response(self, response: httpx.Response) -> OpenAICompatResponse:
        return OpenAICompatResponse.from_http_response(response)
