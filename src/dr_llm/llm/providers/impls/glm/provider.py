from __future__ import annotations

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.transports.openai_compat.provider import (
    OpenAICompatProvider,
)


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
                base_url="https://api.z.ai/api/coding/paas/v4",
                api_key_env=ApiKeyNames.GLM,
            ),
            client=client,
        )
