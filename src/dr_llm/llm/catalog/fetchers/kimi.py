from __future__ import annotations

from datetime import datetime
from typing import Any

from dr_llm.llm.catalog.fetchers.common import (
    as_bool,
    as_int,
    fetch_models_with_template,
    require_api_key,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.kimi_code import KimiCodeProvider
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model


KIMI_CATALOG_URL = "https://api.kimi.com/coding/v1/models"


def fetch_kimi_models(
    provider: KimiCodeProvider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    provider_name = provider.name
    key = require_api_key(
        api_key=provider.config.api_key,
        env_var="KIMI_API_KEY",
        label="Kimi",
    )

    def process(item: dict[str, Any], now: datetime) -> ModelCatalogEntry | None:
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            return None
        return ModelCatalogEntry(
            provider=provider_name,
            model=model_id,
            display_name=str(item.get("display_name") or model_id),
            context_window=as_int(item.get("context_length")),
            max_output_tokens=as_int(item.get("max_output_tokens")),
            supports_reasoning=as_bool(item.get("supports_reasoning")),
            reasoning_capabilities=reasoning_capabilities_for_model(
                provider=provider_name,
                model=model_id,
            ),
            supports_vision=True,
            metadata=item,
            fetched_at=now,
            source_quality="live",
        )

    return fetch_models_with_template(
        url=KIMI_CATALOG_URL,
        headers={"Authorization": f"Bearer {key}"},
        items_key="data",
        item_processor=process,
    )
