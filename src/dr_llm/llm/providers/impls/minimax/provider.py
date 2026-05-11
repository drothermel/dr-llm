from __future__ import annotations

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.anthropic.provider_config import (
    AnthropicProviderConfig,
)
from dr_llm.llm.providers.impls.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.impls.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.impls.minimax.reasoning import MiniMaxReasoningConfig
from dr_llm.llm.request import LlmRequest

MINIMAX_PROVIDER_NAME = ProviderName.MINIMAX
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


class MiniMaxProvider(AnthropicProvider):
    def __init__(
        self,
        config: AnthropicProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or AnthropicProviderConfig(
                name=MINIMAX_PROVIDER_NAME,
                base_url=MINIMAX_BASE_URL,
                api_key_env=MINIMAX_API_KEY_ENV,
            ),
            client=client,
        )

    def _build_request(self, request: LlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            reasoning_mapping=MiniMaxReasoningConfig.from_base(
                request.reasoning
            ),
            require_max_tokens=False,
        )
