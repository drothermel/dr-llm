from __future__ import annotations

import httpx

from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.anthropic.config import AnthropicConfig

MINIMAX_PROVIDER_NAME = "minimax"
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


class MiniMaxAdapter(AnthropicAdapter):
    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or AnthropicConfig(
                name=MINIMAX_PROVIDER_NAME,
                base_url=MINIMAX_BASE_URL,
                api_key_env=MINIMAX_API_KEY_ENV,
            ),
            client=client,
        )
