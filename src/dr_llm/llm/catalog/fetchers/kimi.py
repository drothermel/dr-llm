from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dr_llm.llm.catalog.fetchers.common import api_key_from_env, as_int, get_json
from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model


KIMI_CATALOG_URL = "https://api.kimi.com/coding/v1/models"
KIMI_PROVIDER_NAME = "kimi-code"


def fetch_kimi_models(
    *,
    api_key: str | None = None,
    provider_name: str = KIMI_PROVIDER_NAME,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    key = api_key or api_key_from_env("KIMI_API_KEY")
    if not key:
        raise ProviderSemanticError("Missing KIMI_API_KEY for catalog sync")
    payload = get_json(
        url=KIMI_CATALOG_URL,
        headers={"Authorization": f"Bearer {key}"},
    )
    items_raw = payload.get("data")
    items = items_raw if isinstance(items_raw, list) else []
    now = datetime.now(UTC)
    out: list[ModelCatalogEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        out.append(
            ModelCatalogEntry(
                provider=provider_name,
                model=model_id,
                display_name=str(item.get("display_name") or model_id),
                context_window=as_int(item.get("context_length")),
                max_output_tokens=as_int(item.get("max_output_tokens")),
                supports_reasoning=(
                    bool(item.get("supports_reasoning"))
                    if item.get("supports_reasoning") is not None
                    else None
                ),
                reasoning_capabilities=reasoning_capabilities_for_model(
                    provider=provider_name,
                    model=model_id,
                ),
                supports_vision=True,
                metadata=item,
                fetched_at=now,
                source_quality="live",
            )
        )
    return out, payload
