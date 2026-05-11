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


class GlmUrls(StrEnum):
    API_BASE = "https://api.z.ai/api/coding/paas/v4"
    MODELS_DOCS = "https://docs.z.ai/guides/llm/glm-4.5"


class GlmProvider(OpenAICompatProvider):
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
