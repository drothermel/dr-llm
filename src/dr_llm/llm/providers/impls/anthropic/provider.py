from __future__ import annotations

import httpx

from dr_llm.llm.providers.impls.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.impls.anthropic.reasoning import (
    AnthropicReasoningConfig,
)
from dr_llm.llm.providers.impls.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.impls.anthropic.response import AnthropicResponse
from dr_llm.llm.providers.transports.api_provider import ApiProvider
from dr_llm.llm.request import ApiBackedLlmRequest


class AnthropicProvider(ApiProvider):
    _config: AnthropicConfig

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(config=config or AnthropicConfig(), client=client)

    @property
    def config(self) -> AnthropicConfig:
        return self._config

    def _build_request(self, request: ApiBackedLlmRequest) -> AnthropicRequest:
        return AnthropicRequest.from_llm_request(
            request,
            self._config,
            reasoning_mapping=AnthropicReasoningConfig.from_base(
                request.reasoning
            ),
        )

    def _parse_response(self, response: httpx.Response) -> AnthropicResponse:
        return AnthropicResponse.from_http_response(response)
