from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

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


CatalogFetcher = Callable[[Provider], tuple[list[ModelCatalogEntry], dict[str, Any]]]


def _fetch_minimax_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    return fetch_static_minimax_models(cast(MiniMaxProvider, provider))


def _fetch_openai_compat_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    return fetch_openai_compat_models(cast(OpenAICompatProvider, provider))


def _fetch_kimi_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    kimi_provider = cast(KimiCodeProvider, provider)
    return fetch_kimi_models(
        api_key=kimi_provider.config.api_key,
        provider_name=kimi_provider.name,
    )


def _fetch_anthropic_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    return fetch_anthropic_models(cast(AnthropicProvider, provider))


def _fetch_google_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    return fetch_google_models(cast(GoogleProvider, provider))


def _fetch_headless_provider_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    return fetch_static_headless_models(
        cast(CodexHeadlessProvider | ClaudeHeadlessProvider, provider)
    )


_PROVIDER_FETCHERS: dict[type[Provider], CatalogFetcher] = {
    MiniMaxProvider: _fetch_minimax_provider_models,
    OpenAICompatProvider: _fetch_openai_compat_provider_models,
    KimiCodeProvider: _fetch_kimi_provider_models,
    AnthropicProvider: _fetch_anthropic_provider_models,
    GoogleProvider: _fetch_google_provider_models,
    CodexHeadlessProvider: _fetch_headless_provider_models,
    ClaudeHeadlessProvider: _fetch_headless_provider_models,
}


def fetch_models_for_provider(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    for provider_type, fetcher in _PROVIDER_FETCHERS.items():
        if isinstance(provider, provider_type):
            return fetcher(provider)
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
