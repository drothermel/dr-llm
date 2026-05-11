from __future__ import annotations

from typing import Any

import pytest

from dr_llm.llm import ProviderName
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.impls.anthropic.provider_config import (
    AnthropicProviderConfig,
)
from dr_llm.llm.providers.impls.google.orchestrator import GoogleOrchestrator
from dr_llm.llm.providers.impls.google.provider import GoogleProvider
from dr_llm.llm.providers.impls.kimi_code.orchestrator import (
    KimiCodeOrchestrator,
)
from dr_llm.llm.providers.impls.kimi_code.provider import KimiCodeProvider
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.impls.openai.orchestrator import (
    OpenAIOrchestrator,
)
from dr_llm.llm.providers.impls.openai.provider import OpenAIProvider
from dr_llm.llm.providers.impls.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.provider import OpenRouterProvider


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
        controls_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert controls_fn("gemini-test").supports_reasoning is False
        return expected

    monkeypatch.setattr(
        "dr_llm.llm.providers.impls.google.orchestrator.fetch_google_models",
        fake_fetch_google_models,
    )

    assert orchestrator.fetch_models() == expected


def test_kimi_orchestrator_fetches_with_wrapped_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = KimiCodeProvider(
        config=AnthropicProviderConfig(
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
        controls_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert received_provider.config.api_key == "kimi-secret"
        assert received_provider.name == ProviderName.KIMI_CODE
        assert controls_fn("kimi-for-coding").supports_reasoning is True
        return [], {"source": "kimi"}

    monkeypatch.setattr(
        "dr_llm.llm.providers.impls.kimi_code.orchestrator.fetch_kimi_models",
        fake_fetch_kimi_models,
    )

    assert orchestrator.fetch_models() == ([], {"source": "kimi"})


def test_openai_compat_orchestrator_fetches_with_wrapped_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(
        config=OpenAICompatConfig(
            name=ProviderName.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key="openai-secret",
        )
    )
    orchestrator = OpenAIOrchestrator(provider)

    def fake_fetch_openai_compat_models(
        received_provider: OpenAIProvider,
        *,
        controls_fn,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        assert controls_fn("gpt-5-mini").supports_reasoning is True
        return [], {"source": "openai_compat"}

    monkeypatch.setattr(
        "dr_llm.llm.providers.impls.openai_compat_base.fetch_openai_compat_models",
        fake_fetch_openai_compat_models,
    )

    assert orchestrator.fetch_models() == ([], {"source": "openai_compat"})


def test_openrouter_orchestrator_applies_policy_to_live_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenRouterProvider(
        config=OpenAICompatConfig(
            name=ProviderName.OPENROUTER,
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            api_key="openrouter-secret",
        )
    )
    orchestrator = OpenRouterOrchestrator(provider)
    entries = [
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat-v3.1",
            supports_reasoning=False,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat",
            supports_reasoning=True,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="unknown/model",
            source_quality="live",
        ),
    ]

    def fake_fetch_models(_orchestrator):
        return entries, {"source": "openrouter"}

    monkeypatch.setattr(
        "dr_llm.llm.providers.impls.openrouter.orchestrator."
        "BaseOpenAICompatOrchestrator.fetch_models",
        fake_fetch_models,
    )

    filtered, raw_payload = orchestrator.fetch_models()

    assert raw_payload == {"source": "openrouter"}
    assert [entry.model for entry in filtered] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
    ]
    assert filtered[0].supports_reasoning is True
    assert filtered[1].supports_reasoning is False
    assert "dr_llm_controls" in filtered[0].metadata
