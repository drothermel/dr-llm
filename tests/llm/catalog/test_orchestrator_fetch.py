from __future__ import annotations

from typing import Any

import pytest

from dr_llm.llm import ProviderName
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.google.orchestrator import GoogleOrchestrator
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.kimi_code.orchestrator import KimiCodeOrchestrator
from dr_llm.llm.providers.kimi_code.provider import KimiCodeProvider
from dr_llm.llm.providers.openai_compat_config import OpenAICompatConfig
from dr_llm.llm.providers.openai.orchestrator import (
    OpenAIOrchestrator,
)
from dr_llm.llm.providers.openai_compat_provider import OpenAICompatProvider
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities


class _GoogleSubclassProvider(GoogleProvider):
    pass


def test_google_orchestrator_fetches_with_wrapped_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _GoogleSubclassProvider()
    orchestrator = GoogleOrchestrator(provider)
    expected = (
        [ModelCatalogEntry(provider=provider.name, model="gemini-test")],
        {"source": ProviderName.GOOGLE},
    )

    def fake_fetch_google_models(
        received_provider: GoogleProvider,
        *,
        capabilities_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert capabilities_fn("gemini-test") is None
        return expected

    monkeypatch.setattr(
        "dr_llm.llm.providers.google.orchestrator.fetch_google_models",
        fake_fetch_google_models,
    )

    assert orchestrator.fetch_models() == expected


def test_kimi_orchestrator_fetches_with_wrapped_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = KimiCodeProvider(
        config=AnthropicConfig(
            name=ProviderName.KIMI_CODE,
            base_url="https://api.kimi.com/coding/v1/messages",
            api_key_env="KIMI_API_KEY",
            api_key="kimi-secret",
        )
    )
    orchestrator = KimiCodeOrchestrator(provider)

    def fake_fetch_kimi_models(
        received_provider: KimiCodeProvider,
        *,
        capabilities_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert received_provider.config.api_key == "kimi-secret"
        assert received_provider.name == ProviderName.KIMI_CODE
        assert capabilities_fn("kimi-for-coding") is not None
        return [], {"source": "kimi"}

    monkeypatch.setattr(
        "dr_llm.llm.providers.kimi_code.orchestrator.fetch_kimi_models",
        fake_fetch_kimi_models,
    )

    assert orchestrator.fetch_models() == ([], {"source": "kimi"})


def test_openai_compat_orchestrator_fetches_with_wrapped_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatProvider(
        config=OpenAICompatConfig(
            name=ProviderName.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key="openai-secret",
        )
    )
    orchestrator = OpenAIOrchestrator(provider)

    def fake_fetch_openai_compat_models(
        received_provider: OpenAICompatProvider,
        *,
        capabilities_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert isinstance(capabilities_fn("gpt-5-mini"), ReasoningCapabilities)
        return [], {"source": "openai_compat"}

    monkeypatch.setattr(
        "dr_llm.llm.providers.openai_compat_orchestrator.fetch_openai_compat_models",
        fake_fetch_openai_compat_models,
    )

    assert orchestrator.fetch_models() == ([], {"source": "openai_compat"})
