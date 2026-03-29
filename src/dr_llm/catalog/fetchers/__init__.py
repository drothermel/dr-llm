from __future__ import annotations

from typing import Any

from dr_llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.catalog.fetchers.google import fetch_google_models
from dr_llm.catalog.fetchers.kimi import KIMI_PROVIDER_NAME, fetch_kimi_models
from dr_llm.catalog.fetchers.openai_compat import fetch_openai_compat_models
from dr_llm.catalog.fetchers.static import (
    fetch_static_headless_models,
    fetch_static_minimax_models,
)
from dr_llm.providers.anthropic import AnthropicAdapter
from dr_llm.providers.google import GoogleAdapter
from dr_llm.providers.headless_adapter import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from dr_llm.providers.openai_compat import OpenAICompatAdapter
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.catalog.models import ModelCatalogEntry


def fetch_models_for_adapter(
    adapter: ProviderAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    if isinstance(adapter, OpenAICompatAdapter):
        if adapter.name == "minimax":
            return fetch_static_minimax_models(adapter)
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
