from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.catalog.fetchers.common import (
    as_int,
    fetch_models_with_template,
    require_api_key,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model


def fetch_anthropic_models(
    provider: AnthropicProvider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    parsed = urlsplit(provider.config.base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/messages"):
        raise ProviderSemanticError(
            f"Anthropic base URL must end with '/messages' for catalog sync. Got: {provider.config.base_url}"
        )
    models_path = f"{path[: -len('/messages')]}/models"
    models_url = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
    key = require_api_key(
        api_key=provider.config.api_key,
        env_var=provider.config.api_key_env,
        label="Anthropic",
    )
    headers = {
        "anthropic-version": provider.config.anthropic_version,
        "x-api-key": key,
    }

    def process(item: dict[str, Any], now: datetime) -> ModelCatalogEntry | None:
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            return None
        return ModelCatalogEntry(
            provider=provider.name,
            model=model_id,
            display_name=str(item.get("display_name") or model_id),
            context_window=as_int(item.get("context_window")),
            max_output_tokens=as_int(item.get("max_output_tokens")),
            reasoning_capabilities=reasoning_capabilities_for_model(
                provider=provider.name,
                model=model_id,
            ),
            supports_vision=None,
            metadata=item,
            fetched_at=now,
            source_quality="live",
        )

    return fetch_models_with_template(
        url=models_url,
        headers=headers,
        items_key="data",
        item_processor=process,
    )
