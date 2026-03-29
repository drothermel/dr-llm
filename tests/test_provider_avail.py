from __future__ import annotations

import pytest

from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.anthropic.config import AnthropicConfig
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import (
    ProviderAvailabilityStatus,
    ProviderConfig,
)
from dr_llm.providers.google.adapter import GoogleAdapter
from dr_llm.providers.openai_compat.adapter import OpenAICompatAdapter
from dr_llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode
from dr_llm.providers.usage import TokenUsage


class _FakeAdapter(ProviderAdapter):
    def __init__(
        self,
        name: str,
        *,
        config: ProviderConfig | None = None,
    ) -> None:
        self._config = config or ProviderConfig(name=name)

    def generate(self, request: LlmRequest) -> LlmResponse:  # noqa: ARG002
        return LlmResponse(
            text="ok",
            usage=TokenUsage(),
            provider="fake",
            model="fake",
            mode=CallMode.api,
        )


def test_supported_provider_names_are_sorted_and_canonical() -> None:
    registry = ProviderRegistry()
    registry.register(_FakeAdapter("b-provider"))
    registry.register(_FakeAdapter("A-Provider"))

    assert registry.sorted_names() == ["a-provider", "b-provider"]


def test_adapter_availability_status_reports_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAKE_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.providers.provider_config.shutil.which",
        lambda executable: None if executable == "fake-cli" else "/usr/bin/ok",
    )

    adapter = _FakeAdapter(
        "fake-provider",
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


def test_registry_availability_statuses_report_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAKE_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.providers.provider_config.shutil.which",
        lambda executable: None if executable == "fake-cli" else "/usr/bin/ok",
    )

    registry = ProviderRegistry()
    registry.register(
        _FakeAdapter(
            "fake-provider",
            config=ProviderConfig(
                name="fake-provider",
                required_env_vars=["FAKE_ENV"],
                required_executables=["fake-cli"],
            ),
        )
    )

    statuses = registry.availability_statuses()
    assert len(statuses) == 1
    assert statuses[0].provider == "fake-provider"
    assert statuses[0].available is False
    assert statuses[0].missing_env_vars == ("FAKE_ENV",)
    assert statuses[0].missing_executables == ("fake-cli",)
    assert statuses[0].supports_structured_output is False


def test_registry_available_names_filter_to_available_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("READY_ENV", "present")
    monkeypatch.setattr(
        "dr_llm.providers.provider_config.shutil.which",
        lambda executable: "/usr/bin/ready" if executable == "ready-cli" else None,
    )

    registry = ProviderRegistry()
    registry.register(
        _FakeAdapter(
            "ready-provider",
            config=ProviderConfig(
                name="ready-provider",
                required_env_vars=["READY_ENV"],
                required_executables=["ready-cli"],
            ),
        )
    )
    registry.register(
        _FakeAdapter(
            "missing-provider",
            config=ProviderConfig(
                name="missing-provider",
                required_env_vars=["MISSING_ENV"],
            ),
        )
    )

    assert registry.available_names() == ["ready-provider"]


def test_registry_available_names_uses_precomputed_statuses() -> None:
    registry = ProviderRegistry()
    statuses = [
        ProviderAvailabilityStatus(provider="ready-provider", available=True),
        ProviderAvailabilityStatus(provider="missing-provider", available=False),
    ]

    assert registry.available_names(statuses=statuses) == ["ready-provider"]


def test_openai_compat_inline_api_key_suppresses_env_requirement() -> None:
    adapter = OpenAICompatAdapter(
        config=OpenAICompatConfig(
            name="openai",
            base_url="https://api.example.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key="inline-key",
        ),
    )

    assert adapter.config.required_env_vars == []


def test_anthropic_inline_api_key_suppresses_env_requirement() -> None:
    adapter = AnthropicAdapter(
        config=AnthropicConfig(
            api_key_env="ANTHROPIC_API_KEY",
            api_key="inline-key",
        )
    )

    assert adapter.config.required_env_vars == []
    adapter.close()


def test_google_inline_api_key_suppresses_env_requirement() -> None:
    adapter = GoogleAdapter(
        config=APIProviderConfig(
            name="google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GOOGLE_API_KEY",
            api_key="inline-key",
        )
    )

    assert adapter.config.required_env_vars == []
