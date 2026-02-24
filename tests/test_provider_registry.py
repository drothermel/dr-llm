from __future__ import annotations

import pytest

from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.registry import ProviderRegistry
from llm_pool.types import CallMode, LlmRequest, LlmResponse, TokenUsage


class _FakeAdapter(ProviderAdapter):
    mode = "api"

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def generate(self, request: LlmRequest) -> LlmResponse:  # noqa: ARG002
        return LlmResponse(
            text="ok",
            usage=TokenUsage(),
            provider="fake",
            model="fake",
            mode=CallMode.api,
        )


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


def test_register_rejects_empty_alias() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="provider alias must be non-empty"):
        registry.register(_FakeAdapter(name="fake"), aliases=[""])


def test_register_rejects_whitespace_alias() -> None:
    registry = ProviderRegistry()
    with pytest.raises(
        ValueError, match="provider alias must not have leading or trailing whitespace"
    ):
        registry.register(_FakeAdapter(name="fake"), aliases=[" alias "])


def test_register_normalizes_keys_to_lowercase() -> None:
    registry = ProviderRegistry()
    adapter = _FakeAdapter(name="FakeProvider")
    registry.register(adapter, aliases=["ALIAS"])

    assert registry.get("fakeprovider") is adapter
    assert registry.get("alias") is adapter
