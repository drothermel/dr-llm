from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.catalog.fetchers.kimi import fetch_kimi_models
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


CatalogResult = tuple[list[ModelCatalogEntry], dict[str, Any]]
CatalogFetcher = Callable[[Any], CatalogResult]

# Order matters: KimiCodeProvider is a subclass of AnthropicProvider, so it
# must be checked first. Fetchers are referenced by name so test monkeypatches
# of the module-level symbols are honored.
_PROVIDER_FETCHERS: tuple[tuple[type[Provider], str], ...] = (
    (MiniMaxProvider, "fetch_static_minimax_models"),
    (OpenAICompatProvider, "fetch_openai_compat_models"),
    (KimiCodeProvider, "fetch_kimi_models"),
    (AnthropicProvider, "fetch_anthropic_models"),
    (GoogleProvider, "fetch_google_models"),
    (CodexHeadlessProvider, "fetch_static_headless_models"),
    (ClaudeHeadlessProvider, "fetch_static_headless_models"),
)


def fetch_models_for_provider(provider: Provider) -> CatalogResult:
    for provider_type, fetcher_name in _PROVIDER_FETCHERS:
        if isinstance(provider, provider_type):
            fetcher: CatalogFetcher = globals()[fetcher_name]
            return fetcher(provider)
    return [], {"source": "unsupported_provider_type"}


__all__ = [
    "fetch_anthropic_models",
    "fetch_google_models",
    "fetch_kimi_models",
    "fetch_models_for_provider",
    "fetch_openai_compat_models",
    "fetch_static_headless_models",
    "fetch_static_minimax_models",
]
