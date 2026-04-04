from __future__ import annotations

import httpx

from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.anthropic.config import AnthropicConfig

KIMI_CODE_PROVIDER_NAME = "kimi-code"
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1/messages"
KIMI_CODE_API_KEY_ENV = "KIMI_API_KEY"


class KimiCodeAdapter(AnthropicAdapter):
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
