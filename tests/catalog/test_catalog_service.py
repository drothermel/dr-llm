from __future__ import annotations

from typing import Any

import pytest

from dr_llm.catalog.models import ModelCatalogEntry, ModelCatalogSyncResult
from dr_llm.catalog.service import ModelCatalogService
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import ProviderConfig
from dr_llm.providers.registry import ProviderRegistry
from tests.conftest import make_response


class _DummyAdapter(ProviderAdapter):
    def __init__(self) -> None:
        self._config = ProviderConfig(name="dummy")

    def generate(self, request: LlmRequest) -> LlmResponse:
        return make_response(provider=request.provider, model=request.model)


class _FakeRepo:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self.replaced: dict[str, list[ModelCatalogEntry]] = {}

    def record_model_catalog_snapshot(
        self,
        *,
        provider: str,
        status: str,
        raw_payload: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> str:
        self.snapshots.append(
            {
                "provider": provider,
                "status": status,
                "raw_payload": raw_payload or {},
                "error_text": error_text,
            }
        )
        return f"snap-{len(self.snapshots)}"

    def replace_provider_models(
        self,
        *,
        provider: str,
        entries: list[ModelCatalogEntry],
    ) -> int:
        self.replaced[provider] = entries
        return len(entries)

    def list_models(self, *, query: object) -> list[ModelCatalogEntry]:
        return []

    def count_models(self, *, query: object) -> int:
        return 0

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        return None


def test_sync_writes_snapshots_and_replaces_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ProviderRegistry()
    registry.register(_DummyAdapter())
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def fake_fetch(adapter: Any) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [ModelCatalogEntry(provider=adapter.name, model="dummy-model", source_quality="live")],
            {"data": [{"id": "dummy-model"}]},
        )

    monkeypatch.setattr("dr_llm.catalog.service.fetch_models_for_adapter", fake_fetch)
    monkeypatch.setattr(
        "dr_llm.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )

    results = service.sync_models_detailed(provider="dummy")
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert repo.snapshots
    assert repo.replaced["dummy"][0].model == "dummy-model"


def test_sync_records_failure_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ProviderRegistry()
    registry.register(_DummyAdapter())
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def failing_fetch(adapter: Any) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        raise RuntimeError("network timeout")

    monkeypatch.setattr("dr_llm.catalog.service.fetch_models_for_adapter", failing_fetch)
    monkeypatch.setattr(
        "dr_llm.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )

    results = service.sync_models_detailed(provider="dummy")
    assert len(results) == 1
    assert not results[0].success
    assert "network timeout" in (results[0].error or "")
    assert len(repo.snapshots) == 1
    assert repo.snapshots[0]["status"] == "failed"


def test_sync_filters_blacklisted_models_before_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AnthropicDummyAdapter(_DummyAdapter):
        def __init__(self) -> None:
            self._config = ProviderConfig(name="anthropic")

    registry = ProviderRegistry()
    registry.register(_AnthropicDummyAdapter())
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def fake_fetch(adapter: Any) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="claude-3-haiku-20240307",
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="claude-haiku-4-5-20251001",
                    source_quality="live",
                ),
            ],
            {"data": [{"id": "claude-3-haiku-20240307"}]},
        )

    monkeypatch.setattr("dr_llm.catalog.service.fetch_models_for_adapter", fake_fetch)
    monkeypatch.setattr(
        "dr_llm.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )

    results = service.sync_models_detailed(provider="anthropic")
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert [entry.model for entry in repo.replaced["anthropic"]] == [
        "claude-haiku-4-5-20251001"
    ]


def test_sync_applies_openrouter_policy_filter_and_reasoning_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OpenRouterDummyAdapter(_DummyAdapter):
        def __init__(self) -> None:
            self._config = ProviderConfig(name="openrouter")

    registry = ProviderRegistry()
    registry.register(_OpenRouterDummyAdapter())
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def fake_fetch(adapter: Any) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="deepseek/deepseek-chat-v3.1",
                    supports_reasoning=False,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="deepseek/deepseek-chat",
                    supports_reasoning=True,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="unknown/model",
                    source_quality="live",
                ),
            ],
            {"data": []},
        )

    monkeypatch.setattr("dr_llm.catalog.service.fetch_models_for_adapter", fake_fetch)
    monkeypatch.setattr(
        "dr_llm.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )

    results = service.sync_models_detailed(provider="openrouter")
    assert len(results) == 1
    assert results[0].success
    assert [entry.model for entry in repo.replaced["openrouter"]] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
    ]
    assert repo.replaced["openrouter"][0].supports_reasoning is True
    assert repo.replaced["openrouter"][1].supports_reasoning is False
