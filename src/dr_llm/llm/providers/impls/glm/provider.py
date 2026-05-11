from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.impls.glm.controls import GlmReasoningConfig
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


class GlmUrls(StrEnum):
    API_BASE = "https://api.z.ai/api/coding/paas/v4"
    MODELS_DOCS = "https://docs.z.ai/guides/llm/glm-4.5"


class GlmProvider(ApiProvider):
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
                name=ProviderName.GLM,
                base_url=GlmUrls.API_BASE,
                api_key_env=ApiKeyNames.GLM,
            ),
            client=client,
        )

    @property
    def config(self) -> OpenAICompatConfig:
        return self._config

    def _build_request(self, request: LlmRequest) -> OpenAICompatRequest:
        controls = GlmReasoningConfig.from_base(request.reasoning)
        return OpenAICompatRequest.from_llm_request(
            request,
            self._config,
            extra_body=controls.extra_body,
            warnings=controls.warnings,
        )

    def _parse_response(
        self, response: httpx.Response
    ) -> OpenAICompatResponse:
        return OpenAICompatResponse.from_http_response(response)
