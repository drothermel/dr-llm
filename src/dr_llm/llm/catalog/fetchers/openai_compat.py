from __future__ import annotations

from datetime import datetime
from typing import Any

from dr_llm.llm.catalog.fetchers.common import (
    api_key_from_env,
    fetch_models_with_template,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry, ModelCatalogPricing
from dr_llm.llm.coercion import as_float, as_int
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.providers.reasoning_capabilities import (
    ReasoningCapabilities,
    reasoning_capabilities_for_model,
)


def fetch_openai_compat_models(
    provider: OpenAICompatProvider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    base = provider.config.base_url.rstrip("/")
    endpoint = f"{base}/models"
    key = provider.config.api_key or api_key_from_env(provider.config.api_key_env)
    headers: dict[str, str] | None = None
    if key:
        headers = {"Authorization": f"Bearer {key}"}

    def process(item: dict[str, Any], now: datetime) -> ModelCatalogEntry | None:
        return _process_openai_model_item(
            item=item, now=now, provider_name=provider.name
        )

    return fetch_models_with_template(
        url=endpoint,
        headers=headers,
        items_key="data",
        item_processor=process,
    )


def _process_openai_model_item(
    *,
    item: dict[str, Any],
    now: datetime,
    provider_name: str,
) -> ModelCatalogEntry | None:
    model_id = str(item.get("id") or item.get("name") or "").strip()
    if not model_id:
        return None
    pricing = _parse_pricing(item.get("pricing"))
    reasoning_capabilities = reasoning_capabilities_for_model(
        provider=provider_name,
        model=model_id,
    )
    supports_reasoning, reasoning_capabilities = _resolve_reasoning_support(
        supported_params=item.get("supported_parameters"),
        reasoning_capabilities=reasoning_capabilities,
    )
    return ModelCatalogEntry(
        provider=provider_name,
        model=model_id,
        display_name=str(item.get("name") or model_id),
        context_window=as_int(item.get("context_length")),
        max_output_tokens=as_int(item.get("max_output_tokens")),
        supports_reasoning=supports_reasoning,
        reasoning_capabilities=reasoning_capabilities,
        supports_vision=_detect_supports_vision(item),
        pricing=pricing,
        metadata=item,
        fetched_at=now,
        source_quality="live",
    )


def _resolve_reasoning_support(
    *,
    supported_params: Any,
    reasoning_capabilities: ReasoningCapabilities | None,
) -> tuple[bool | None, ReasoningCapabilities | None]:
    if not isinstance(supported_params, list):
        return None, reasoning_capabilities
    normalized = {str(param) for param in supported_params}
    supports_reasoning = "reasoning" in normalized or "reasoning.effort" in normalized
    if supports_reasoning and reasoning_capabilities is None:
        reasoning_capabilities = ReasoningCapabilities(mode="openai_effort")
    return supports_reasoning, reasoning_capabilities


def _parse_pricing(value: Any) -> ModelCatalogPricing | None:
    if not isinstance(value, dict):
        return None
    input_cost = (
        as_float(value.get("prompt"))
        if value.get("prompt") is not None
        else as_float(value.get("input"))
    )
    output_cost = (
        as_float(value.get("completion"))
        if value.get("completion") is not None
        else as_float(value.get("output"))
    )
    reasoning_cost = as_float(value.get("reasoning"))
    if input_cost is None and output_cost is None and reasoning_cost is None:
        return None
    # Most OpenRouter/OpenAI-compatible metadata expresses price per-token.
    # Normalize to per-1M tokens for catalog consistency.
    scale = 1_000_000.0
    return ModelCatalogPricing(
        currency="USD",
        input_cost_per_1m=input_cost * scale if input_cost is not None else None,
        output_cost_per_1m=output_cost * scale if output_cost is not None else None,
        reasoning_cost_per_1m=(
            reasoning_cost * scale if reasoning_cost is not None else None
        ),
        raw=value,
    )


def _detect_supports_vision(item: dict[str, Any]) -> bool | None:
    modalities = item.get("input_modalities")
    if isinstance(modalities, list):
        return any(
            str(modality).lower() in {"image", "vision"} for modality in modalities
        )
    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        mods = architecture.get("input_modalities")
        if isinstance(mods, list):
            return any(
                str(modality).lower() in {"image", "vision"} for modality in mods
            )
    return None
