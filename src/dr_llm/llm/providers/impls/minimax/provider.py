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
from dr_llm.llm.providers.impls.minimax.controls import MiniMaxControlMapping
from dr_llm.llm.request import LlmRequest


class MiniMaxUrls(StrEnum):
    MESSAGES_API = "https://api.minimax.io/anthropic/v1/messages"
    MODELS_DOCS = "https://platform.minimax.io/docs/guides/models-intro"


class MiniMaxProvider(AnthropicProvider):
    def __init__(
        self,
        config: AnthropicProviderConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or AnthropicProviderConfig(
                name=ProviderName.MINIMAX,
                base_url=MiniMaxUrls.MESSAGES_API,
                api_key_env=ApiKeyNames.MINIMAX,
            ),
            client=client,
        )

    def _build_request(self, request: LlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            control_mapping=MiniMaxControlMapping.from_base(request.reasoning),
            require_max_tokens=False,
        )
