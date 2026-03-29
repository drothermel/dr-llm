from __future__ import annotations

import pytest

from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.anthropic.config import AnthropicConfig
from dr_llm.providers.google.adapter import GoogleAdapter
from dr_llm.providers.openai_compat.adapter import OpenAICompatAdapter
from dr_llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.providers.provider_config import ProviderAvailabilityStatus, ProviderConfig
from dr_llm.providers.registry import ProviderRegistry
from tests.conftest import FakeAdapter


def test_availability_reports_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAKE_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.providers.provider_config.shutil.which",
        lambda exe: None if exe == "fake-cli" else "/usr/bin/ok",
    )

    adapter = FakeAdapter(
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
        "dr_llm.providers.provider_config.shutil.which",
        lambda exe: "/usr/bin/ready" if exe == "ready-cli" else None,
    )

    registry = ProviderRegistry()
    registry.register(
        FakeAdapter(
            name="ready-provider",
            config=ProviderConfig(
                name="ready-provider",
                required_env_vars=["READY_ENV"],
                required_executables=["ready-cli"],
            ),
        )
    )
    registry.register(
        FakeAdapter(
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
        lambda: OpenAICompatAdapter(
            config=OpenAICompatConfig(
                name="openai",
                base_url="https://api.example.com/v1",
                api_key_env="OPENAI_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: AnthropicAdapter(
            config=AnthropicConfig(
                api_key_env="ANTHROPIC_API_KEY",
                api_key="inline-key",
            )
        ),
        lambda: GoogleAdapter(
            config=APIProviderConfig(
                name="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_env="GOOGLE_API_KEY",
                api_key="inline-key",
            )
        ),
    ],
    ids=["openai_compat", "anthropic", "google"],
)
def test_inline_api_key_suppresses_env_requirement(adapter_factory: object) -> None:
    adapter = adapter_factory()  # type: ignore[operator]
    try:
        assert adapter.config.required_env_vars == []
    finally:
        adapter.close()
