from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Protocol

from dr_llm.errors import PersistenceError
from dr_llm.llm.catalog.models import (
    ModelCatalogEntry,
    ModelCatalogQuery,
    ModelCatalogSyncResult,
)
from dr_llm.llm.catalog.model_blacklist import filter_blacklisted_entries
from dr_llm.llm.catalog.fetchers import fetch_models_for_provider
from dr_llm.llm.providers.openrouter.policy import apply_openrouter_model_policies
from dr_llm.llm.providers.registry import ProviderRegistry

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

    async def sync_models_detailed(
        self,
        *,
        provider: str | None = None,
    ) -> list[ModelCatalogSyncResult]:
        targets = self._resolve_targets(provider=provider)
        return await self._sync_targets_in_parallel(targets)

    async def _sync_targets_in_parallel(
        self, targets: list[str]
    ) -> list[ModelCatalogSyncResult]:
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(self._sync_one_provider, target)
                    for target in targets
                )
            )
        )

    def _sync_one_provider(self, target: str) -> ModelCatalogSyncResult:
        try:
            entries, raw_payload = self._fetch_provider(target)
            entries = apply_openrouter_model_policies(
                filter_blacklisted_entries(entries)
            )
            return self._record_sync_success(target, entries, raw_payload)
        except Exception as exc:  # noqa: BLE001
            return self._record_sync_failure(target, exc)

    def _record_sync_success(
        self,
        target: str,
        entries: list[ModelCatalogEntry],
        raw_payload: dict[str, Any],
    ) -> ModelCatalogSyncResult:
        snapshot_id: str | None = None
        if self._repository is not None:
            self._repository.replace_provider_models(
                provider=target,
                entries=entries,
            )
            snapshot_id = self._repository.record_model_catalog_snapshot(
                provider=target,
                status="success",
                raw_payload=raw_payload,
            )
        return ModelCatalogSyncResult(
            provider=target,
            success=True,
            entry_count=len(entries),
            snapshot_id=snapshot_id,
            raw_payload=raw_payload,
        )

    def _record_sync_failure(
        self, target: str, exc: BaseException
    ) -> ModelCatalogSyncResult:
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
        return ModelCatalogSyncResult(
            provider=target,
            success=False,
            entry_count=0,
            snapshot_id=snapshot_id,
            error=detailed_error,
        )

    def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        return self._require_repository().list_models(query=query)

    def count_models(self, query: ModelCatalogQuery) -> int:
        return self._require_repository().count_models(query=query)

    def show_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        return self._require_repository().get_model(provider=provider, model=model)

    def _require_repository(self) -> ModelCatalogRepository:
        if self._repository is None:
            raise PersistenceError("ModelCatalogService.repository is not configured")
        return self._repository

    def _resolve_targets(self, *, provider: str | None) -> list[str]:
        if provider is not None:
            return [self._registry.get(provider).name]
        return sorted(
            {self._registry.get(name).name for name in self._registry.names()}
        )

    def _fetch_provider(
        self, provider: str
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return fetch_models_for_provider(self._registry.get(provider))
