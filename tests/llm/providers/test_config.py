from __future__ import annotations

import pytest

from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.kimi_code import KimiCodeProvider
from dr_llm.llm.providers.minimax import MiniMaxProvider
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.config import ProviderAvailabilityStatus, ProviderConfig
from dr_llm.llm.providers.registry import ProviderRegistry
from tests.conftest import FakeProvider


def test_availability_reports_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAKE_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.llm.providers.config.shutil.which",
        lambda exe: None if exe == "fake-cli" else "/usr/bin/ok",
    )

    adapter = FakeProvider(
        name="fake-provider",
        config=ProviderConfig(
            name="fake-provider",
            required_env_vars=["FAKE_ENV"],
            required_executables=["fake-cli"],
        ),
    )

    status = adapter.availability_status()
    assert status.provider == "fake-provider"
    assert status.available is False
    assert status.missing_env_vars == ("FAKE_ENV",)
    assert status.missing_executables == ("fake-cli",)
    assert status.supports_structured_output is False


def test_registry_available_names_filters_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("READY_ENV", "present")
    monkeypatch.delenv("MISSING_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.llm.providers.config.shutil.which",
        lambda exe: "/usr/bin/ready" if exe == "ready-cli" else None,
    )

    registry = ProviderRegistry()
    registry.register(
        FakeProvider(
            name="ready-provider",
            config=ProviderConfig(
                name="ready-provider",
                required_env_vars=["READY_ENV"],
                required_executables=["ready-cli"],
            ),
        )
    )
    registry.register(
        FakeProvider(
            name="missing-provider",
            config=ProviderConfig(
                name="missing-provider",
                required_env_vars=["MISSING_ENV"],
            ),
        )
    )

    assert registry.available_names() == ["ready-provider"]


def test_available_names_accepts_precomputed_statuses() -> None:
    registry = ProviderRegistry()
    statuses = [
        ProviderAvailabilityStatus(provider="ready-provider", available=True),
        ProviderAvailabilityStatus(provider="missing-provider", available=False),
    ]

    assert registry.available_names(statuses=statuses) == ["ready-provider"]


@pytest.mark.parametrize(
    "adapter_factory",
    [
        lambda: OpenAICompatProvider(
            config=OpenAICompatConfig(
                name="openai",
                base_url="https://api.example.com/v1",
                api_key_env="OPENAI_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: AnthropicProvider(
            config=AnthropicConfig(
                api_key_env="ANTHROPIC_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: GoogleProvider(
            config=APIProviderConfig(
                name="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_env="GOOGLE_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: KimiCodeProvider(
            config=AnthropicConfig(
                name="kimi-code",
                base_url="https://api.kimi.com/coding/v1/messages",
                api_key_env="KIMI_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: MiniMaxProvider(
            config=AnthropicConfig(
                name="minimax",
                base_url="https://api.minimax.io/anthropic/v1/messages",
                api_key_env="MINIMAX_API_KEY",
                api_key="inline-key",
            )
        ),
    ],
    ids=["openai_compat", "anthropic", "google", "kimi_code", "minimax"],
)
def test_inline_api_key_suppresses_env_requirement(adapter_factory: object) -> None:
    adapter = adapter_factory()  # type: ignore[operator]
    try:
        assert adapter.config.required_env_vars == []
    finally:
        adapter.close()
