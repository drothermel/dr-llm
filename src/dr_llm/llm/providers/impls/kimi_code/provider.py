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
from dr_llm.llm.providers.impls.kimi_code.reasoning import (
    KimiCodeReasoningConfig,
)
from dr_llm.llm.request import LlmRequest

KIMI_CODE_PROVIDER_NAME = ProviderName.KIMI_CODE
KIMI_CODE_API_KEY_ENV = ApiKeyNames.KIMI


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
                name=KIMI_CODE_PROVIDER_NAME,
                base_url=KimiCodeUrls.MESSAGES_API,
                api_key_env=KIMI_CODE_API_KEY_ENV,
            ),
            client=client,
        )

    def _build_request(self, request: LlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            reasoning_mapping=KimiCodeReasoningConfig.from_base(
                request.reasoning
            ),
        )
