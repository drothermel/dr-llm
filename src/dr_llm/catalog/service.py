from __future__ import annotations

import logging
import traceback
from typing import Any, Protocol

from dr_llm.catalog.models import (
    ModelCatalogEntry,
    ModelCatalogQuery,
    ModelCatalogSyncResult,
)
from dr_llm.catalog.model_blacklist import filter_blacklisted_entries
from dr_llm.catalog.fetchers import (
    fetch_models_for_adapter,
    fetch_out_of_registry_provider_models,
)
from dr_llm.catalog.fetchers.kimi import KIMI_PROVIDER_NAME
from dr_llm.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class ModelCatalogRepository(Protocol):
    def record_model_catalog_snapshot(
        self,
        *,
        provider: str,
        status: str,
        raw_payload: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> str: ...

    def replace_provider_models(
        self,
        *,
        provider: str,
        entries: list[ModelCatalogEntry],
    ) -> int: ...

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]: ...

    def count_models(self, *, query: ModelCatalogQuery) -> int: ...

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None: ...


class ModelCatalogService:
    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        repository: ModelCatalogRepository | None = None,
    ) -> None:
        self._registry = registry
        self._repository = repository

    def sync_models(self, *, provider: str | None = None) -> dict[str, int]:
        results = self.sync_models_detailed(provider=provider)
        return {result.provider: result.entry_count for result in results}

    def sync_models_detailed(
        self,
        *,
        provider: str | None = None,
    ) -> list[ModelCatalogSyncResult]:
        targets = self._resolve_targets(provider=provider)
        results: list[ModelCatalogSyncResult] = []
        for target in targets:
            try:
                entries, raw_payload = self._fetch_provider(target)
                entries = filter_blacklisted_entries(entries)
                snapshot_id: str | None = None
                if self._repository is not None:
                    snapshot_id = self._repository.record_model_catalog_snapshot(
                        provider=target,
                        status="success",
                        raw_payload=raw_payload,
                    )
                    self._repository.replace_provider_models(
                        provider=target,
                        entries=entries,
                    )
                results.append(
                    ModelCatalogSyncResult(
                        provider=target,
                        success=True,
                        entry_count=len(entries),
                        snapshot_id=snapshot_id,
                        raw_payload=raw_payload,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.exception("Model catalog sync failed provider=%s", target)
                detailed_error = f"{exc}\n{traceback.format_exc()}"
                snapshot_id: str | None = None
                if self._repository is not None:
                    snapshot_id = self._repository.record_model_catalog_snapshot(
                        provider=target,
                        status="failed",
                        raw_payload={},
                        error_text=detailed_error,
                    )
                results.append(
                    ModelCatalogSyncResult(
                        provider=target,
                        success=False,
                        entry_count=0,
                        snapshot_id=snapshot_id,
                        error=detailed_error,
                    )
                )
        return results

    def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        if self._repository is None:
            return []
        return self._repository.list_models(query=query)

    def count_models(self, query: ModelCatalogQuery) -> int:
        if self._repository is None:
            return 0
        return self._repository.count_models(query=query)

    def show_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        if self._repository is None:
            return None
        return self._repository.get_model(provider=provider, model=model)

    def _resolve_targets(self, *, provider: str | None) -> list[str]:
        if provider is not None:
            if provider == KIMI_PROVIDER_NAME:
                return [KIMI_PROVIDER_NAME]
            adapter = self._registry.get(provider)
            return [adapter.name]
        canonical: set[str] = set()
        for name in self._registry.names():
            canonical.add(self._registry.get(name).name)
        canonical.add(KIMI_PROVIDER_NAME)
        return sorted(canonical)

    def _fetch_provider(
        self, provider: str
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        if provider == KIMI_PROVIDER_NAME:
            return fetch_out_of_registry_provider_models(provider)
        adapter = self._registry.get(provider)
        return fetch_models_for_adapter(adapter)
