from __future__ import annotations

from dr_llm.llm import ProviderName
import asyncio
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.catalog.service import ModelCatalogService
from dr_llm.llm import (
    ProviderRegistry,
)
from tests.conftest import FakeOrchestrator


class _FakeRepo:
    def __init__(self) -> None:
        self.actions: list[str] = []
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
        self.actions.append("snapshot")
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
        self.actions.append("replace")
        self.replaced[provider] = entries
        return len(entries)

    def list_models(self, *, query: object) -> list[ModelCatalogEntry]:
        return []

    def count_models(self, *, query: object) -> int:
        return 0

    def get_model(
        self, *, provider: str, model: str
    ) -> ModelCatalogEntry | None:
        return None


def test_sync_writes_snapshots_and_replaces_models() -> None:
    def fake_fetch() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider="dummy",
                    model="dummy-model",
                    source_quality="live",
                )
            ],
            {"data": [{"id": "dummy-model"}]},
        )

    registry = ProviderRegistry()
    registry.register(
        FakeOrchestrator(name="dummy", fetch_models_fn=fake_fetch)
    )
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    results = asyncio.run(service.sync_models_detailed(provider="dummy"))
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert repo.actions == ["replace", "snapshot"]
    assert repo.snapshots
    assert repo.replaced["dummy"][0].model == "dummy-model"


def test_sync_records_failure_on_fetch_error() -> None:
    def failing_fetch() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        raise RuntimeError("network timeout")

    registry = ProviderRegistry()
    registry.register(
        FakeOrchestrator(name="dummy", fetch_models_fn=failing_fetch)
    )
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    results = asyncio.run(service.sync_models_detailed(provider="dummy"))
    assert len(results) == 1
    assert not results[0].success
    assert "network timeout" in (results[0].error or "")
    assert len(repo.snapshots) == 1
    assert repo.snapshots[0]["status"] == "failed"


def test_sync_filters_blacklisted_models_before_replace() -> None:
    def fake_fetch() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider=ProviderName.ANTHROPIC,
                    model="claude-3-haiku-20240307",
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=ProviderName.ANTHROPIC,
                    model="claude-haiku-4-5-20251001",
                    source_quality="live",
                ),
            ],
            {"data": [{"id": "claude-3-haiku-20240307"}]},
        )

    registry = ProviderRegistry()
    registry.register(
        FakeOrchestrator(
            name=ProviderName.ANTHROPIC, fetch_models_fn=fake_fetch
        )
    )
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    results = asyncio.run(
        service.sync_models_detailed(provider=ProviderName.ANTHROPIC)
    )
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert [
        entry.model for entry in repo.replaced[ProviderName.ANTHROPIC]
    ] == ["claude-haiku-4-5-20251001"]


def test_sync_applies_openrouter_policy_filter_and_reasoning_metadata() -> (
    None
):
    def fake_fetch() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider=ProviderName.OPENROUTER,
                    model="deepseek/deepseek-chat-v3.1",
                    supports_reasoning=False,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=ProviderName.OPENROUTER,
                    model="deepseek/deepseek-chat",
                    supports_reasoning=True,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider=ProviderName.OPENROUTER,
                    model="unknown/model",
                    source_quality="live",
                ),
            ],
            {"data": []},
        )

    registry = ProviderRegistry()
    registry.register(
        FakeOrchestrator(
            name=ProviderName.OPENROUTER, fetch_models_fn=fake_fetch
        )
    )
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    results = asyncio.run(
        service.sync_models_detailed(provider=ProviderName.OPENROUTER)
    )
    assert len(results) == 1
    assert results[0].success
    assert [
        entry.model for entry in repo.replaced[ProviderName.OPENROUTER]
    ] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
    ]
    assert repo.replaced[ProviderName.OPENROUTER][0].supports_reasoning is True
    assert (
        repo.replaced[ProviderName.OPENROUTER][1].supports_reasoning is False
    )


def test_sync_records_failure_when_replace_fails_without_success_snapshot() -> (
    None
):
    class _ReplaceFailsRepo(_FakeRepo):
        def replace_provider_models(
            self,
            *,
            provider: str,
            entries: list[ModelCatalogEntry],
        ) -> int:
            self.actions.append("replace")
            raise RuntimeError(
                f"replace failed for {provider} with {len(entries)} entries"
            )

    def fake_fetch() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [
                ModelCatalogEntry(
                    provider="dummy",
                    model="dummy-model",
                    source_quality="live",
                )
            ],
            {"data": [{"id": "dummy-model"}]},
        )

    registry = ProviderRegistry()
    registry.register(
        FakeOrchestrator(name="dummy", fetch_models_fn=fake_fetch)
    )
    repo = _ReplaceFailsRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    results = asyncio.run(service.sync_models_detailed(provider="dummy"))

    assert len(results) == 1
    assert not results[0].success
    assert "replace failed for dummy with 1 entries" in (
        results[0].error or ""
    )
    assert repo.actions == ["replace", "snapshot"]
    assert repo.snapshots == [
        {
            "provider": "dummy",
            "status": "failed",
            "raw_payload": {},
            "error_text": results[0].error,
        }
    ]
