from __future__ import annotations

from enum import StrEnum

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.transports.api_config import APIProviderConfig


class AnthropicUrls(StrEnum):
    MESSAGES_API = "https://api.anthropic.com/v1/messages"
    MODELS_DOCS = "https://docs.anthropic.com/en/docs/about-claude/models"


class AnthropicProviderConfig(APIProviderConfig):
    name: ProviderName = ProviderName.ANTHROPIC
    base_url: str = AnthropicUrls.MESSAGES_API
    api_key_env: str = ApiKeyNames.ANTHROPIC
    anthropic_version: str = "2023-06-01"
