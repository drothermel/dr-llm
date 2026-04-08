from __future__ import annotations

import pytest

from dr_llm.llm.providers.registry import ProviderRegistry
from tests.conftest import FakeProvider


def test_register_rejects_empty_name() -> None:
    registry = ProviderRegistry()
    adapter = FakeProvider(name="")
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(adapter)


def test_register_rejects_whitespace_name() -> None:
    registry = ProviderRegistry()
    adapter = FakeProvider(name="  fake  ")
    with pytest.raises(ValueError, match="whitespace"):
        registry.register(adapter)


def test_normalizes_keys_to_lowercase() -> None:
    registry = ProviderRegistry()
    registry.register(FakeProvider(name="FakeProvider"))
    assert registry.get("fakeprovider") is not None


def test_rejects_duplicate_normalized_name() -> None:
    registry = ProviderRegistry()
    registry.register(FakeProvider(name="fake"))
    with pytest.raises(ValueError, match="conflict"):
        registry.register(FakeProvider(name="FAKE"))


def test_get_unknown_raises_key_error() -> None:
    registry = ProviderRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_close_releases_adapters(fake_provider: FakeProvider) -> None:
    registry = ProviderRegistry()
    registry.register(fake_provider)
    registry.close()
    assert fake_provider.close_calls == 1
    assert registry.names() == set()


def test_sorted_names() -> None:
    registry = ProviderRegistry()
    registry.register(FakeProvider(name="zebra"))
    registry.register(FakeProvider(name="alpha"))
    registry.register(FakeProvider(name="mango"))
    assert registry.sorted_names() == ["alpha", "mango", "zebra"]
