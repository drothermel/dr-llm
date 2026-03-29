from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dr_llm.catalog.fetchers.common import api_key_from_env, as_int, get_json
from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.catalog.models import ModelCatalogEntry


def fetch_anthropic_models(
    adapter: AnthropicAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    parsed = urlsplit(adapter.config.base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/messages"):
        raise ProviderSemanticError(
            f"Anthropic base URL must end with '/messages' for catalog sync. Got: {adapter.config.base_url}"
        )
    models_path = f"{path[: -len('/messages')]}/models"
    models_url = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
    key = adapter.config.api_key or api_key_from_env(adapter.config.api_key_env)
    if not key:
        raise ProviderSemanticError(
            f"Missing Anthropic API key for catalog sync. Set {adapter.config.api_key_env}"
        )
    headers = {
        "anthropic-version": adapter.config.anthropic_version,
        "x-api-key": key,
    }
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
                context_window=as_int(item.get("context_window")),
                max_output_tokens=as_int(item.get("max_output_tokens")),
                supports_reasoning=True,
                supports_vision=None,
                metadata=item,
                fetched_at=now,
                source_quality="live",
            )
        )
    return out, payload
