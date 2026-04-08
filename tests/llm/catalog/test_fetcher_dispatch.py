from __future__ import annotations

from typing import Any

import pytest

from dr_llm.llm.catalog.fetchers import fetch_models_for_provider
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.providers.config import ProviderConfig
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.kimi_code import KimiCodeProvider
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from tests.conftest import make_response


class _UnsupportedProvider(Provider):
    def __init__(self) -> None:
        self._config = ProviderConfig(name="unsupported")

    def generate(self, request: LlmRequest) -> LlmResponse:
        return make_response(provider=request.provider, model=request.model)


class _GoogleSubclassProvider(GoogleProvider):
    pass


def test_fetch_models_for_provider_dispatches_google_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _GoogleSubclassProvider()
    expected = (
        [ModelCatalogEntry(provider=provider.name, model="gemini-test")],
        {"source": "google"},
    )

    def fake_fetch_google_models(
        received_provider: GoogleProvider,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        return expected

    monkeypatch.setattr(
        "dr_llm.llm.catalog.fetchers.fetch_google_models",
        fake_fetch_google_models,
    )

    assert fetch_models_for_provider(provider) == expected


def test_fetch_models_for_provider_passes_kimi_config_to_fetcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = KimiCodeProvider(
        config=AnthropicConfig(
            name="kimi-code",
            base_url="https://api.kimi.com/coding/v1/messages",
            api_key_env="KIMI_API_KEY",
            api_key="kimi-secret",
        )
    )

    def fake_fetch_kimi_models(
        *,
        api_key: str | None = None,
        provider_name: str = "",
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert api_key == "kimi-secret"
        assert provider_name == "kimi-code"
        return [], {"source": "kimi"}

    monkeypatch.setattr(
        "dr_llm.llm.catalog.fetchers.fetch_kimi_models",
        fake_fetch_kimi_models,
    )

    assert fetch_models_for_provider(provider) == ([], {"source": "kimi"})


def test_fetch_models_for_provider_dispatches_openai_compat_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OpenAICompatSubclassProvider(OpenAICompatProvider):
        pass

    provider = _OpenAICompatSubclassProvider(
        config=OpenAICompatConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key="openai-secret",
        )
    )

    def fake_fetch_openai_compat_models(
        received_provider: OpenAICompatProvider,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        assert received_provider is provider
        return [], {"source": "openai_compat"}

    monkeypatch.setattr(
        "dr_llm.llm.catalog.fetchers.fetch_openai_compat_models",
        fake_fetch_openai_compat_models,
    )

    assert fetch_models_for_provider(provider) == ([], {"source": "openai_compat"})


def test_fetch_models_for_provider_returns_unsupported_for_unknown_provider() -> None:
    assert fetch_models_for_provider(_UnsupportedProvider()) == (
        [],
        {"source": "unsupported_provider_type"},
    )
