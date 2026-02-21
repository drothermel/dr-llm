from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_pool.catalog.fetchers.common import api_key_from_env, get_json
from llm_pool.errors import ProviderSemanticError
from llm_pool.providers.google import GoogleAdapter
from llm_pool.types import ModelCatalogEntry


def fetch_google_models(
    adapter: GoogleAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    key = adapter._config.api_key or api_key_from_env(adapter._config.api_key_env)  # noqa: SLF001
    if not key:
        raise ProviderSemanticError(
            f"Missing Google API key for catalog sync. Set {adapter._config.api_key_env}"  # noqa: SLF001
        )
    endpoint = f"{adapter._config.base_url.rstrip('/')}/models?key={key}"  # noqa: SLF001
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
        supports_tools = _first_bool(
            item,
            "supportsTools",
            "supports_tools",
            "toolCalling",
            "tool_calling",
            "functionCalling",
            "function_calling",
        )
        if supports_tools is None:
            supports_tools = True
        supports_vision = _first_bool(
            item,
            "supportsVision",
            "supports_vision",
            "vision",
            "multimodal",
        )
        if supports_vision is None:
            supports_vision = True
        out.append(
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_name,
                display_name=str(item.get("displayName") or model_name),
                context_window=_as_int(item.get("inputTokenLimit")),
                max_output_tokens=_as_int(item.get("outputTokenLimit")),
                supports_reasoning=supports_reasoning,
                supports_tools=supports_tools,
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
