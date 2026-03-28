from __future__ import annotations

import pytest

from dr_llm.providers.anthropic import AnthropicAdapter, AnthropicConfig
from dr_llm.providers.avail import (
    ProviderAvailabilityStatus,
    available_provider_names,
    supported_provider_names,
    supported_provider_statuses,
)
from dr_llm.providers.base import (
    ProviderAdapter,
    ProviderCapabilities,
    ProviderRuntimeRequirements,
)
from dr_llm.providers.google import GoogleAdapter, GoogleConfig
from dr_llm.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.types import CallMode, LlmRequest, LlmResponse, TokenUsage


class _FakeAdapter(ProviderAdapter):
    mode = "api"

    def __init__(
        self,
        name: str,
        *,
        requirements: ProviderRuntimeRequirements | None = None,
    ) -> None:
        self.name = name
        self._requirements = requirements or ProviderRuntimeRequirements()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_native_tools=True,
            supports_structured_output=False,
        )

    @property
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        return self._requirements

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

    assert supported_provider_names(registry) == ["a-provider", "b-provider"]


def test_supported_provider_statuses_report_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAKE_ENV", raising=False)
    monkeypatch.setattr(
        "dr_llm.providers.avail.shutil.which",
        lambda executable: None if executable == "fake-cli" else "/usr/bin/ok",
    )

    registry = ProviderRegistry()
    registry.register(
        _FakeAdapter(
            "fake-provider",
            requirements=ProviderRuntimeRequirements(
                required_env_vars=["FAKE_ENV"],
                required_executables=["fake-cli"],
            ),
        )
    )

    statuses = supported_provider_statuses(registry)
    assert len(statuses) == 1
    assert statuses[0].provider == "fake-provider"
    assert statuses[0].available is False
    assert statuses[0].missing_env_vars == ("FAKE_ENV",)
    assert statuses[0].missing_executables == ("fake-cli",)
    assert statuses[0].supports_native_tools is True
    assert statuses[0].supports_structured_output is False


def test_available_provider_names_filter_to_available_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("READY_ENV", "present")
    monkeypatch.setattr(
        "dr_llm.providers.avail.shutil.which",
        lambda executable: "/usr/bin/ready" if executable == "ready-cli" else None,
    )

    registry = ProviderRegistry()
    registry.register(
        _FakeAdapter(
            "ready-provider",
            requirements=ProviderRuntimeRequirements(
                required_env_vars=["READY_ENV"],
                required_executables=["ready-cli"],
            ),
        )
    )
    registry.register(
        _FakeAdapter(
            "missing-provider",
            requirements=ProviderRuntimeRequirements(
                required_env_vars=["MISSING_ENV"],
            ),
        )
    )

    assert available_provider_names(registry) == ["ready-provider"]


def test_available_provider_names_uses_precomputed_statuses() -> None:
    registry = ProviderRegistry()
    statuses = [
        ProviderAvailabilityStatus(provider="ready-provider", available=True),
        ProviderAvailabilityStatus(provider="missing-provider", available=False),
    ]

    assert available_provider_names(registry, statuses=statuses) == ["ready-provider"]



def test_openai_compat_inline_api_key_suppresses_env_requirement() -> None:
    adapter = OpenAICompatAdapter(
        name="openai",
        config=OpenAICompatConfig(
            base_url="https://api.example.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key="inline-key",
        ),
    )

    assert adapter.runtime_requirements.required_env_vars == []


def test_anthropic_inline_api_key_suppresses_env_requirement() -> None:
    adapter = AnthropicAdapter(
        config=AnthropicConfig(
            api_key_env="ANTHROPIC_API_KEY",
            api_key="inline-key",
        )
    )

    assert adapter.runtime_requirements.required_env_vars == []
    adapter.close()


def test_google_inline_api_key_suppresses_env_requirement() -> None:
    adapter = GoogleAdapter(
        config=GoogleConfig(
            api_key_env="GOOGLE_API_KEY",
            api_key="inline-key",
        )
    )

    assert adapter.runtime_requirements.required_env_vars == []
