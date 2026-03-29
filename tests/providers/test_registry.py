from __future__ import annotations

import pytest

from dr_llm.providers.registry import ProviderRegistry
from tests.conftest import FakeAdapter


def test_register_rejects_empty_name() -> None:
    registry = ProviderRegistry()
    adapter = FakeAdapter(name="")
    with pytest.raises(ValueError):
        registry.register(adapter)


def test_register_rejects_whitespace_name() -> None:
    registry = ProviderRegistry()
    adapter = FakeAdapter(name="  fake  ")
    with pytest.raises(ValueError):
        registry.register(adapter)


def test_normalizes_keys_to_lowercase() -> None:
    registry = ProviderRegistry()
    registry.register(FakeAdapter(name="FakeProvider"))
    assert registry.get("fakeprovider") is not None


def test_rejects_duplicate_normalized_name() -> None:
    registry = ProviderRegistry()
    registry.register(FakeAdapter(name="fake"))
    with pytest.raises(ValueError):
        registry.register(FakeAdapter(name="FAKE"))


def test_get_unknown_raises_key_error() -> None:
    registry = ProviderRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_close_releases_adapters() -> None:
    registry = ProviderRegistry()
    adapter = FakeAdapter(name="fake")
    registry.register(adapter)
    registry.close()
    assert adapter.close_calls == 1
    assert registry.names() == set()


def test_sorted_names() -> None:
    registry = ProviderRegistry()
    registry.register(FakeAdapter(name="zebra"))
    registry.register(FakeAdapter(name="alpha"))
    registry.register(FakeAdapter(name="mango"))
    assert registry.sorted_names() == ["alpha", "mango", "zebra"]
