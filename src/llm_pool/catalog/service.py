from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Protocol

import yaml

from llm_pool.catalog.fetchers import (
    fetch_models_for_adapter,
    fetch_out_of_registry_provider_models,
)
from llm_pool.catalog.fetchers.kimi import KIMI_PROVIDER_NAME
from llm_pool.catalog.models import (
    DEFAULT_MODEL_OVERRIDES_PATH,
    ModelCatalogSyncResult,
    ModelOverridesFile,
)
from llm_pool.providers.registry import ProviderRegistry
from llm_pool.types import ModelCatalogEntry, ModelCatalogQuery

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

    def upsert_model_overrides(self, *, entries: list[ModelCatalogEntry]) -> int: ...

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]: ...

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None: ...


class ModelCatalogService:
    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        repository: ModelCatalogRepository | None = None,
        overlays_path: Path | None = None,
    ) -> None:
        self._registry = registry
        self._repository = repository
        self._overlays_path = overlays_path or DEFAULT_MODEL_OVERRIDES_PATH

    def sync_models(self, *, provider: str | None = None) -> dict[str, int]:
        results = self.sync_models_detailed(provider=provider)
        return {result.provider: result.entry_count for result in results}

    def sync_models_detailed(
        self,
        *,
        provider: str | None = None,
    ) -> list[ModelCatalogSyncResult]:
        targets = self._resolve_targets(provider=provider)
        overlays = self._load_overrides()
        overlay_by_provider = _group_by_provider(overlays)
        results: list[ModelCatalogSyncResult] = []
        persisted_overrides: list[ModelCatalogEntry] = []
        for target in targets:
            try:
                live_entries, raw_payload = self._fetch_provider(target)
                merged = merge_overlay_entries(
                    live_entries=live_entries,
                    overlays=overlay_by_provider.get(target, []),
                )
                snapshot_id: str | None = None
                if self._repository is not None:
                    snapshot_id = self._repository.record_model_catalog_snapshot(
                        provider=target,
                        status="success",
                        raw_payload=raw_payload,
                    )
                    self._repository.replace_provider_models(
                        provider=target,
                        entries=merged,
                    )
                persisted_overrides.extend(overlay_by_provider.get(target, []))
                results.append(
                    ModelCatalogSyncResult(
                        provider=target,
                        success=True,
                        entry_count=len(merged),
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
        if self._repository is not None and persisted_overrides:
            self._repository.upsert_model_overrides(entries=persisted_overrides)
        return results

    def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        if self._repository is None:
            return []
        return self._repository.list_models(query=query)

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

    def _load_overrides(self) -> list[ModelCatalogEntry]:
        if not self._overlays_path.exists():
            return []
        text = self._overlays_path.read_text(encoding="utf-8")
        payload: dict[str, Any]
        try:
            parsed = json.loads(text)
            payload = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            try:
                loaded = yaml.safe_load(text)
                payload = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError as exc:
                snippet = text[:200].replace("\n", "\\n")
                logger.warning(
                    "Failed to parse model overrides as YAML: %s snippet=%s",
                    exc,
                    snippet,
                )
                payload = {}
            except Exception as exc:  # noqa: BLE001
                snippet = text[:200].replace("\n", "\\n")
                logger.warning(
                    "Failed to parse model overrides as YAML: %s snippet=%s",
                    exc,
                    snippet,
                )
                payload = {}
        model = ModelOverridesFile(**payload)
        return [
            entry.model_copy(update={"source_quality": "overlay"})
            for entry in model.models
        ]


def _group_by_provider(
    entries: list[ModelCatalogEntry],
) -> dict[str, list[ModelCatalogEntry]]:
    out: dict[str, list[ModelCatalogEntry]] = {}
    for entry in entries:
        out.setdefault(entry.provider, []).append(entry)
    return out


def merge_overlay_entries(
    *,
    live_entries: list[ModelCatalogEntry],
    overlays: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    by_key: dict[tuple[str, str], ModelCatalogEntry] = {
        (entry.provider, entry.model): entry for entry in live_entries
    }
    for overlay in overlays:
        key = (overlay.provider, overlay.model)
        base = by_key.get(key)
        if base is None:
            by_key[key] = overlay.model_copy(update={"source_quality": "overlay"})
            continue
        updated = base.model_copy(
            update={
                "pricing": overlay.pricing or base.pricing,
                "rate_limits": overlay.rate_limits or base.rate_limits,
                "source_quality": "overlay",
                "metadata": {**base.metadata, **overlay.metadata},
                "display_name": overlay.display_name or base.display_name,
                "context_window": (
                    overlay.context_window
                    if overlay.context_window is not None
                    else base.context_window
                ),
                "max_output_tokens": (
                    overlay.max_output_tokens
                    if overlay.max_output_tokens is not None
                    else base.max_output_tokens
                ),
                "supports_reasoning": (
                    overlay.supports_reasoning
                    if overlay.supports_reasoning is not None
                    else base.supports_reasoning
                ),
                "supports_tools": (
                    overlay.supports_tools
                    if overlay.supports_tools is not None
                    else base.supports_tools
                ),
                "supports_vision": (
                    overlay.supports_vision
                    if overlay.supports_vision is not None
                    else base.supports_vision
                ),
            }
        )
        by_key[key] = updated
    return sorted(by_key.values(), key=lambda entry: (entry.provider, entry.model))
