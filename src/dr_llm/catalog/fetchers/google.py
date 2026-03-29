from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dr_llm.catalog.fetchers.common import api_key_from_env, get_json
from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.google.adapter import GoogleAdapter
from dr_llm.catalog.models import ModelCatalogEntry


def fetch_google_models(
    adapter: GoogleAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    key = adapter.config.api_key or api_key_from_env(adapter.config.api_key_env)
    if not key:
        raise ProviderSemanticError(
            f"Missing Google API key for catalog sync. Set {adapter.config.api_key_env}"
        )
    endpoint = f"{adapter.config.base_url.rstrip('/')}/models?key={key}"
    payload = get_json(url=endpoint)
    items_raw = payload.get("models")
    items = items_raw if isinstance(items_raw, list) else []
    now = datetime.now(timezone.utc)
    out: list[ModelCatalogEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name_raw = str(item.get("name") or "").strip()
        if not name_raw:
            continue
        model_name = name_raw.split("/")[-1]
        supports_reasoning = bool(item.get("thinking")) if "thinking" in item else None
        supports_vision = _first_bool(
            item,
            "supportsVision",
            "supports_vision",
            "vision",
            "multimodal",
        )
        out.append(
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_name,
                display_name=str(item.get("displayName") or model_name),
                context_window=_as_int(item.get("inputTokenLimit")),
                max_output_tokens=_as_int(item.get("outputTokenLimit")),
                supports_reasoning=supports_reasoning,
                supports_vision=supports_vision,
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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _first_bool(item: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in item:
            parsed = _as_bool(item.get(key))
            if parsed is not None:
                return parsed
    return None
