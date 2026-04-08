from __future__ import annotations

from datetime import datetime
from typing import Any

from dr_llm.llm.catalog.fetchers.common import (
    as_bool,
    fetch_models_with_template,
    require_api_key,
    resolve_supports_vision,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.coercion import as_int
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model


def fetch_google_models(
    provider: GoogleProvider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    key = require_api_key(
        api_key=provider.config.api_key,
        env_var=provider.config.api_key_env,
        label="Google",
    )
    endpoint = f"{provider.config.base_url.rstrip('/')}/models"

    def process(item: dict[str, Any], now: datetime) -> ModelCatalogEntry | None:
        name_raw = str(item.get("name") or "").strip()
        if not name_raw:
            return None
        model_name = name_raw.split("/")[-1]
        supports_reasoning = (
            as_bool(item.get("thinking")) if "thinking" in item else None
        )
        supports_vision = resolve_supports_vision(
            item,
            "supportsVision",
            "supports_vision",
            "vision",
            "multimodal",
        )
        return ModelCatalogEntry(
            provider=provider.name,
            model=model_name,
            display_name=str(item.get("displayName") or model_name),
            context_window=as_int(item.get("inputTokenLimit")),
            max_output_tokens=as_int(item.get("outputTokenLimit")),
            supports_reasoning=supports_reasoning,
            reasoning_capabilities=reasoning_capabilities_for_model(
                provider=provider.name,
                model=model_name,
            ),
            supports_vision=supports_vision,
            metadata=item,
            fetched_at=now,
            source_quality="live",
        )

    return fetch_models_with_template(
        url=endpoint,
        headers={"x-goog-api-key": key},
        items_key="models",
        item_processor=process,
    )
