from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.core.base import ProviderTransport
from dr_llm.llm.providers.core.controls import ProviderControls


def display_str(model_id: str) -> str:
    return model_id.replace("-", " ").title()


def build_static_catalog_entries(
    *,
    provider: ProviderTransport,
    models: Sequence[str],
    docs_url: str,
    supports_vision: bool | None,
    controls_fn: Callable[[str], ProviderControls],
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    source_meta = {"source": "static", "docs_url": docs_url}
    now = datetime.now(UTC)
    entries = []
    for model_id in models:
        controls = controls_fn(model_id)
        entries.append(
            ModelCatalogEntry(
                provider=provider.name,
                model=model_id,
                display_name=display_str(model_id),
                supports_reasoning=controls.supports_reasoning,
                supports_vision=supports_vision,
                source_quality="static",
                fetched_at=now,
                metadata={
                    **source_meta,
                    "dr_llm_controls": controls.catalog_metadata,
                },
            )
        )
    return entries, source_meta
