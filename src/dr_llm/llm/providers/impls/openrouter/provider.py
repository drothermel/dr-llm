from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.impls.openrouter.request_controls import (
    OpenRouterRequestControls,
)
from dr_llm.llm.providers.transports.api_provider import ApiProvider
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.transports.openai_compat.request import (
    OpenAICompatRequest,
)
from dr_llm.llm.providers.transports.openai_compat.response import (
    OpenAICompatResponse,
)
from dr_llm.llm.request import LlmRequest


class OpenRouterUrls(StrEnum):
    API_BASE = "https://openrouter.ai/api/v1"
    MODELS_DOCS = "https://openrouter.ai/models"


class OpenRouterProvider(ApiProvider):
    _config: OpenAICompatConfig

    def __init__(
        self,
        *,
        config: OpenAICompatConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or OpenAICompatConfig(
                name=ProviderName.OPENROUTER,
                base_url=OpenRouterUrls.API_BASE,
                api_key_env=ApiKeyNames.OPENROUTER,
            ),
            client=client,
        )

    @property
    def config(self) -> OpenAICompatConfig:
        return self._config

    def _build_request(self, request: LlmRequest) -> OpenAICompatRequest:
        request_controls = OpenRouterRequestControls.from_reasoning(
            request.reasoning
        )
        return OpenAICompatRequest.from_llm_request(
            request,
            self._config,
            reasoning_effort=request_controls.reasoning_effort,
            extra_body=request_controls.extra_body,
            warnings=request_controls.warnings,
        )

    def _parse_response(
        self, response: httpx.Response
    ) -> OpenAICompatResponse:
        return OpenAICompatResponse.from_http_response(response)
