from __future__ import annotations

from typing import Any

from llm_pool.catalog.fetchers.anthropic import fetch_anthropic_models
from llm_pool.catalog.fetchers.google import fetch_google_models
from llm_pool.catalog.fetchers.kimi import KIMI_PROVIDER_NAME, fetch_kimi_models
from llm_pool.catalog.fetchers.openai_compat import fetch_openai_compat_models
from llm_pool.catalog.fetchers.static import fetch_static_headless_models
from llm_pool.providers.anthropic import AnthropicAdapter
from llm_pool.providers.base import ProviderAdapter
from llm_pool.providers.google import GoogleAdapter
from llm_pool.providers.headless import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from llm_pool.providers.openai_compat import OpenAICompatAdapter
from llm_pool.types import ModelCatalogEntry


def fetch_models_for_adapter(
    adapter: ProviderAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    if isinstance(adapter, OpenAICompatAdapter):
        return fetch_openai_compat_models(adapter)
    if isinstance(adapter, AnthropicAdapter):
        return fetch_anthropic_models(adapter)
    if isinstance(adapter, GoogleAdapter):
        return fetch_google_models(adapter)
    if isinstance(
        adapter,
        (
            CodexHeadlessAdapter,
            ClaudeHeadlessAdapter,
            ClaudeHeadlessMiniMaxAdapter,
            ClaudeHeadlessKimiAdapter,
        ),
    ):
        return fetch_static_headless_models(adapter)
    return [], {"source": "unsupported_adapter_type"}


def fetch_out_of_registry_provider_models(
    provider: str,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    if provider == KIMI_PROVIDER_NAME:
        return fetch_kimi_models()
    return [], {"source": "unsupported_provider"}


__all__ = [
    "fetch_models_for_adapter",
    "fetch_out_of_registry_provider_models",
]
