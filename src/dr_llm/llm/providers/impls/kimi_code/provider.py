from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.impls.anthropic.provider_config import (
    AnthropicProviderConfig,
)
from dr_llm.llm.providers.impls.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.impls.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.impls.kimi_code.request_controls import (
    KimiCodeRequestControls,
)
from dr_llm.llm.request import LlmRequest


class KimiCodeUrls(StrEnum):
    MESSAGES_API = "https://api.kimi.com/coding/v1/messages"
    MODELS_API = "https://api.kimi.com/coding/v1/models"
    MODELS_DOCS = "https://platform.moonshot.ai/docs/guide/agent/coding"


class KimiCodeProvider(AnthropicProvider):
    def __init__(
        self,
        config: AnthropicProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or AnthropicProviderConfig(
                name=ProviderName.KIMI_CODE,
                base_url=KimiCodeUrls.MESSAGES_API,
                api_key_env=ApiKeyNames.KIMI,
            ),
            client=client,
        )

    def _build_request(self, request: LlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            request_controls=KimiCodeRequestControls.from_reasoning(
                request.reasoning
            ),
        )
