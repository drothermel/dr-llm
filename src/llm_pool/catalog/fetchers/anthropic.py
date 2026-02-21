from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_pool.catalog.fetchers.common import api_key_from_env, get_json
from llm_pool.providers.anthropic import AnthropicAdapter
from llm_pool.types import ModelCatalogEntry


def fetch_anthropic_models(
    adapter: AnthropicAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    base_url = adapter._config.base_url  # noqa: SLF001
    models_url = base_url.replace("/messages", "/models")
    key = adapter._config.api_key or api_key_from_env(adapter._config.api_key_env)  # noqa: SLF001
    headers = {
        "anthropic-version": adapter._config.anthropic_version,  # noqa: SLF001
    }
    if key:
        headers["x-api-key"] = key
    payload = get_json(url=models_url, headers=headers)
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
                provider=adapter.name,
                model=model_id,
                display_name=str(item.get("display_name") or model_id),
                context_window=_as_int(item.get("context_window")),
                max_output_tokens=_as_int(item.get("max_output_tokens")),
                supports_reasoning=True,
                supports_tools=True,
                supports_vision=None,
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
