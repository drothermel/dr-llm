from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_pool.catalog.fetchers.common import api_key_from_env, get_json
from llm_pool.errors import ProviderSemanticError
from llm_pool.types import ModelCatalogEntry


KIMI_CATALOG_URL = "https://api.kimi.com/coding/v1/models"
KIMI_PROVIDER_NAME = "kimi"


def fetch_kimi_models() -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    key = api_key_from_env("KIMI_API_KEY")
    if not key:
        raise ProviderSemanticError("Missing KIMI_API_KEY for catalog sync")
    payload = get_json(
        url=KIMI_CATALOG_URL,
        headers={"Authorization": f"Bearer {key}"},
    )
    items_raw = payload.get("data")
    items = items_raw if isinstance(items_raw, list) else []
    now = datetime.now(timezone.utc)
    out: list[ModelCatalogEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        out.append(
            ModelCatalogEntry(
                provider=KIMI_PROVIDER_NAME,
                model=model_id,
                display_name=str(item.get("display_name") or model_id),
                context_window=_as_int(item.get("context_length")),
                max_output_tokens=_as_int(item.get("max_output_tokens")),
                supports_reasoning=(
                    bool(item.get("supports_reasoning"))
                    if item.get("supports_reasoning") is not None
                    else None
                ),
                supports_tools=True,
                supports_vision=True,
                metadata=item,
                fetched_at=now,
                source_quality="live",
            )
        )
    return out, payload


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None
