from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.transports.openai_compat.provider import (
    OpenAICompatProvider,
)


class OpenRouterUrls(StrEnum):
    API_BASE = "https://openrouter.ai/api/v1"
    MODELS_DOCS = "https://openrouter.ai/models"


class OpenRouterProvider(OpenAICompatProvider):
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
