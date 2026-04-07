from __future__ import annotations

from typing import Any

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.catalog.fetchers.kimi import KIMI_PROVIDER_NAME, fetch_kimi_models
from dr_llm.llm.catalog.fetchers.openai_compat import fetch_openai_compat_models
from dr_llm.llm.catalog.fetchers.static import (
    fetch_static_headless_models,
    fetch_static_minimax_models,
)
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.headless.claude import ClaudeHeadlessProvider
from dr_llm.llm.providers.headless.codex import CodexHeadlessProvider
from dr_llm.llm.providers.kimi_code import KimiCodeProvider
from dr_llm.llm.providers.minimax import MiniMaxProvider
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.catalog.models import ModelCatalogEntry


def fetch_models_for_provider(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    if isinstance(provider, MiniMaxProvider):
        return fetch_static_minimax_models(provider)
    if isinstance(provider, OpenAICompatProvider):
        return fetch_openai_compat_models(provider)
    if isinstance(provider, KimiCodeProvider):
        return fetch_kimi_models(
            api_key=provider.config.api_key,
            provider_name=provider.name,
        )
    if isinstance(provider, AnthropicProvider):
        return fetch_anthropic_models(provider)
    if isinstance(provider, GoogleProvider):
        return fetch_google_models(provider)
    if isinstance(
        provider,
        (
            CodexHeadlessProvider,
            ClaudeHeadlessProvider,
        ),
    ):
        return fetch_static_headless_models(provider)
    return [], {"source": "unsupported_provider_type"}


def fetch_out_of_registry_provider_models(
    provider: str,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    if provider == KIMI_PROVIDER_NAME:
        return fetch_kimi_models()
    return [], {"source": "unsupported_provider"}


__all__ = [
    "fetch_models_for_provider",
    "fetch_out_of_registry_provider_models",
]
