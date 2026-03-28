from __future__ import annotations

from typing import Any

from dr_llm.catalog.service import ModelCatalogService
from dr_llm.providers.base import ProviderAdapter, ProviderRuntimeRequirements
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ModelCatalogEntry,
    TokenUsage,
)


class _DummyAdapter(ProviderAdapter):
    mode = "api"

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        return ProviderRuntimeRequirements()

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

    def list_models(self, *, query) -> list[ModelCatalogEntry]:  # noqa: ANN001
        _ = query
        return []

    def count_models(self, *, query) -> int:  # noqa: ANN001
        _ = query
        return 0

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        _ = (provider, model)
        return None


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
