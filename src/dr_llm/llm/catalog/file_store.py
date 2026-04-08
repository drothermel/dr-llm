from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from dr_llm.errors import PersistenceError
from dr_llm.llm.catalog.model_blacklist import filter_blacklisted_entries
from dr_llm.llm.catalog.models import ModelCatalogEntry, ModelCatalogQuery
from dr_llm.llm.providers.openrouter.policy import apply_openrouter_model_policies

_DEFAULT_CACHE_DIR = Path.home() / ".dr_llm" / "catalog_cache"

logger = logging.getLogger(__name__)


class CatalogCacheCorruptError(PersistenceError):
    """Raised when a catalog cache file exists but cannot be parsed."""


class FileCatalogStore:
    """File-based implementation of ModelCatalogRepository protocol."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR

    def record_model_catalog_snapshot(
        self,
        *,
        provider: str,  # noqa: ARG002
        status: str,  # noqa: ARG002
        raw_payload: dict[str, Any] | None = None,  # noqa: ARG002
        error_text: str | None = None,  # noqa: ARG002
    ) -> str:
        return uuid4().hex

    def replace_provider_models(
        self,
        *,
        provider: str,
        entries: list[ModelCatalogEntry],
    ) -> int:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        data = [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
        target = self._cache_dir / f"{provider}.json"
        fd, tmp_path = tempfile.mkstemp(
            dir=self._cache_dir, suffix=".tmp", prefix=f"{provider}_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            Path(tmp_path).replace(target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return len(entries)

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        entries = self._filtered_entries(query)
        return entries[query.offset : query.offset + query.limit]

    def count_models(self, *, query: ModelCatalogQuery) -> int:
        return len(self._filtered_entries(query))

    def _filtered_entries(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        entries = self._load_all(provider_filter=query.provider)
        return self._apply_filters(entries, query)

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        entries = self._load_provider(provider)
        for entry in entries:
            if entry.model == model:
                return entry
        return None

    @staticmethod
    def _load_one_path(path: Path) -> list[ModelCatalogEntry]:
        """Load a single cache file. Raises ``CatalogCacheCorruptError`` on parse failure."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chunk = [ModelCatalogEntry(**item) for item in data]
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise CatalogCacheCorruptError(
                f"corrupt catalog cache {path}: {exc}"
            ) from exc
        return apply_openrouter_model_policies(filter_blacklisted_entries(chunk))

    def _load_provider(self, provider: str) -> list[ModelCatalogEntry]:
        path = self._cache_dir / f"{provider}.json"
        if not path.exists():
            return []
        return self._load_one_path(path)

    def _load_all(
        self, *, provider_filter: str | None = None
    ) -> list[ModelCatalogEntry]:
        if not self._cache_dir.exists():
            return []
        if provider_filter is not None:
            return self._load_provider(provider_filter)
        entries: list[ModelCatalogEntry] = []
        for path in sorted(self._cache_dir.glob("*.json")):
            try:
                entries.extend(self._load_one_path(path))
            except CatalogCacheCorruptError as exc:
                logger.warning(
                    "Skipping unreadable catalog cache file %s: %s",
                    path,
                    exc,
                )
        return entries

    @staticmethod
    def _apply_filters(
        entries: list[ModelCatalogEntry], query: ModelCatalogQuery
    ) -> list[ModelCatalogEntry]:
        if query.supports_reasoning is not None:
            entries = [
                e for e in entries if e.supports_reasoning == query.supports_reasoning
            ]
        if query.model_contains is not None:
            needle = query.model_contains.lower()
            entries = [e for e in entries if needle in e.model.lower()]
        return entries
