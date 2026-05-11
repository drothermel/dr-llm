from __future__ import annotations

from enum import StrEnum

import httpx

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.impls.openai.families import (
    OPENAI_THINKING_SUPPORTED_MODELS,
)
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.transports.openai_compat.provider import (
    OpenAICompatProvider,
)


class OpenAIUrls(StrEnum):
    API_BASE = "https://api.openai.com/v1"
    MODELS_DOCS = "https://platform.openai.com/docs/models"


class OpenAIProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        config: OpenAICompatConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=config
            or OpenAICompatConfig(
                name=ProviderName.OPENAI,
                base_url=OpenAIUrls.API_BASE,
                api_key_env=ApiKeyNames.OPENAI,
                max_completion_token_model_prefixes=tuple(
                    str(family) for family in OPENAI_THINKING_SUPPORTED_MODELS
                ),
            ),
            client=client,
        )
