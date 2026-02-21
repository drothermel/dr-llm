from __future__ import annotations

from typing import Any

from llm_pool.catalog.service import ModelCatalogService, merge_overlay_entries
from llm_pool.providers.base import ProviderAdapter
from llm_pool.providers.registry import ProviderRegistry
from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ModelCatalogEntry,
    ModelCatalogPricing,
    TokenUsage,
)


class _DummyAdapter(ProviderAdapter):
    mode = "api"

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, request: LlmRequest) -> LlmResponse:  # pragma: no cover - unused
        return LlmResponse(
            text="",
            usage=TokenUsage(),
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
        )


class _FakeRepo:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self.replaced: dict[str, list[ModelCatalogEntry]] = {}
        self.overrides: list[ModelCatalogEntry] = []

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

    def upsert_model_overrides(self, *, entries: list[ModelCatalogEntry]) -> int:
        self.overrides.extend(entries)
        return len(entries)

    def list_models(self, *, query) -> list[ModelCatalogEntry]:  # noqa: ANN001
        _ = query
        return []

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        _ = (provider, model)
        return None


def test_merge_overlay_entries_applies_pricing_override() -> None:
    live = [
        ModelCatalogEntry(
            provider="openrouter",
            model="m1",
            source_quality="live",
            pricing=ModelCatalogPricing(input_cost_per_1m=1.0, output_cost_per_1m=2.0),
        )
    ]
    overlays = [
        ModelCatalogEntry(
            provider="openrouter",
            model="m1",
            source_quality="overlay",
            pricing=ModelCatalogPricing(input_cost_per_1m=3.0, output_cost_per_1m=4.0),
        )
    ]
    merged = merge_overlay_entries(live_entries=live, overlays=overlays)
    assert len(merged) == 1
    assert merged[0].source_quality == "overlay"
    assert merged[0].pricing is not None
    assert merged[0].pricing.input_cost_per_1m == 3.0


def test_catalog_service_sync_writes_snapshots(monkeypatch) -> None:  # noqa: ANN001
    registry = ProviderRegistry()
    registry.register(_DummyAdapter("dummy"))
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def fake_fetch(adapter):  # noqa: ANN001
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="dummy-model",
                    source_quality="live",
                )
            ],
            {"data": [{"id": "dummy-model"}]},
        )

    monkeypatch.setattr("llm_pool.catalog.service.fetch_models_for_adapter", fake_fetch)
    monkeypatch.setattr(
        "llm_pool.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )
    monkeypatch.setattr(
        "llm_pool.catalog.service.ModelCatalogService._load_overrides",
        lambda self: [],
    )

    results = service.sync_models_detailed(provider="dummy")
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert repo.snapshots
    assert repo.replaced["dummy"][0].model == "dummy-model"
