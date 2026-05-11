from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.core.base import ProviderTransport
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities


def build_static_catalog_entries(
    *,
    provider: ProviderTransport,
    models: list[tuple[str, str]],
    docs_url: str,
    supports_vision: bool | None,
    capabilities_fn: Callable[[str], ReasoningCapabilities | None],
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    source_meta = {"source": "static", "docs_url": docs_url}
    now = datetime.now(UTC)
    entries = [
        ModelCatalogEntry(
            provider=provider.name,
            model=model_id,
            display_name=display_name,
            reasoning_capabilities=capabilities_fn(model_id),
            supports_vision=supports_vision,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in models
    ]
    return entries, source_meta
