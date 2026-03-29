from __future__ import annotations

import pytest

from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import ProviderConfig
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.generation.models import CallMode, LlmRequest, LlmResponse, TokenUsage


class _FakeAdapter(ProviderAdapter):
    def __init__(self, name: str) -> None:
        self._config = ProviderConfig(name=name)
        self.close_calls = 0

    def generate(self, request: LlmRequest) -> LlmResponse:  # noqa: ARG002
        return LlmResponse(
            text="ok",
            usage=TokenUsage(),
            provider="fake",
            model="fake",
            mode=CallMode.api,
        )

    def close(self) -> None:
        self.close_calls += 1


def test_register_rejects_empty_adapter_name() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match=r"adapter\.name must be non-empty"):
        registry.register(_FakeAdapter(name=""))


def test_register_rejects_whitespace_adapter_name() -> None:
    registry = ProviderRegistry()
    with pytest.raises(
        ValueError,
        match=r"adapter\.name must not have leading or trailing whitespace",
    ):
        registry.register(_FakeAdapter(name=" fake "))


def test_register_normalizes_keys_to_lowercase() -> None:
    registry = ProviderRegistry()
    adapter = _FakeAdapter(name="FakeProvider")
    registry.register(adapter)

    assert registry.get("fakeprovider") is adapter
    assert registry.names() == {"fakeprovider"}


def test_register_rejects_duplicate_normalized_name() -> None:
    registry = ProviderRegistry()
    registry.register(_FakeAdapter(name="FakeProvider"))

    with pytest.raises(
        ValueError, match=r"register conflict for provider 'fakeprovider'"
    ):
        registry.register(_FakeAdapter(name="fakeprovider"))


def test_alias_lookup_is_not_supported() -> None:
    registry = ProviderRegistry()
    adapter = _FakeAdapter(name="FakeProvider")
    registry.register(adapter)

    with pytest.raises(KeyError, match="Unknown provider"):
        registry.get("alias")


def test_close_releases_registered_adapters() -> None:
    registry = ProviderRegistry()
    adapter = _FakeAdapter(name="FakeProvider")
    registry.register(adapter)

    registry.close()

    assert adapter.close_calls == 1
    assert registry.names() == set()
