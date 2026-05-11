from __future__ import annotations

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.kimi_code.reasoning import KimiCodeReasoningConfig
from dr_llm.llm.request import ApiBackedLlmRequest

KIMI_CODE_PROVIDER_NAME = ProviderName.KIMI_CODE
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1/messages"
KIMI_CODE_API_KEY_ENV = "KIMI_API_KEY"


class KimiCodeProvider(AnthropicProvider):
    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or AnthropicConfig(
                name=KIMI_CODE_PROVIDER_NAME,
                base_url=KIMI_CODE_BASE_URL,
                api_key_env=KIMI_CODE_API_KEY_ENV,
            ),
            client=client,
        )

    def _build_request(self, request: ApiBackedLlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            reasoning_mapping=KimiCodeReasoningConfig.from_base(
                request.reasoning
            ),
        )
