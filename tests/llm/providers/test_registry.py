from __future__ import annotations

import pytest

from dr_llm.llm.providers.core.registry import ProviderRegistry
from tests.conftest import FakeOrchestrator


def test_register_rejects_empty_name() -> None:
    registry = ProviderRegistry()
    orchestrator = FakeOrchestrator(name="")
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(orchestrator)


def test_register_rejects_whitespace_name() -> None:
    registry = ProviderRegistry()
    adapter = FakeOrchestrator(name="  fake  ")
    with pytest.raises(ValueError, match="whitespace"):
        registry.register(adapter)


def test_normalizes_keys_to_lowercase() -> None:
    registry = ProviderRegistry()
    registry.register(FakeOrchestrator(name="FakeProvider"))
    assert registry.get("fakeprovider") is not None


def test_rejects_duplicate_normalized_name() -> None:
    registry = ProviderRegistry()
    registry.register(FakeOrchestrator(name="fake"))
    with pytest.raises(ValueError, match="conflict"):
        registry.register(FakeOrchestrator(name="FAKE"))


def test_get_unknown_raises_key_error() -> None:
    registry = ProviderRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_close_releases_orchestrators(
    fake_orchestrator: FakeOrchestrator,
) -> None:
    registry = ProviderRegistry()
    registry.register(fake_orchestrator)
    registry.close()
    assert fake_orchestrator.close_calls == 1
    assert registry.names() == set()


def test_sorted_names() -> None:
    registry = ProviderRegistry()
    registry.register(FakeOrchestrator(name="zebra"))
    registry.register(FakeOrchestrator(name="alpha"))
    registry.register(FakeOrchestrator(name="mango"))
    assert registry.sorted_names() == ["alpha", "mango", "zebra"]
