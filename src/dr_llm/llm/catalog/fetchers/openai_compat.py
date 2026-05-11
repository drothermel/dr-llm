from __future__ import annotations

from datetime import datetime
from typing import Any

from collections.abc import Callable

from dr_llm.llm.catalog.fetchers.common import (
    api_key_from_env,
    fetch_models_with_template,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry, ModelCatalogPricing
from dr_llm.llm.coercion import as_float, as_int
from dr_llm.llm.providers.core.controls import ProviderControls
from dr_llm.llm.providers.transports.api_provider import ApiProvider

ControlsFn = Callable[[str], ProviderControls]


def fetch_openai_compat_models(
    provider: ApiProvider,
    *,
    controls_fn: ControlsFn,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    base = provider.config.base_url.rstrip("/")
    endpoint = f"{base}/models"
    key = provider.config.api_key or api_key_from_env(
        provider.config.api_key_env
    )
    headers: dict[str, str] | None = None
    if key:
        headers = {"Authorization": f"Bearer {key}"}

    def process(
        item: dict[str, Any], now: datetime
    ) -> ModelCatalogEntry | None:
        return _process_openai_model_item(
            item=item,
            now=now,
            provider_name=provider.name,
            controls_fn=controls_fn,
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
    controls_fn: ControlsFn,
) -> ModelCatalogEntry | None:
    model_id = str(item.get("id") or item.get("name") or "").strip()
    if not model_id:
        return None
    pricing = _parse_pricing(item.get("pricing"))
    controls = controls_fn(model_id)
    supports_reasoning = _resolve_reasoning_support(
        supported_params=item.get("supported_parameters"),
        controls=controls,
    )
    return ModelCatalogEntry(
        provider=provider_name,
        model=model_id,
        display_name=str(item.get("name") or model_id),
        context_window=as_int(item.get("context_length")),
        max_output_tokens=as_int(item.get("max_output_tokens")),
        supports_reasoning=supports_reasoning,
        supports_vision=_detect_supports_vision(item),
        pricing=pricing,
        metadata={
            **item,
            "dr_llm_controls": controls.catalog_metadata,
        },
        fetched_at=now,
        source_quality="live",
    )


def _resolve_reasoning_support(
    *,
    supported_params: Any,
    controls: ProviderControls,
) -> bool | None:
    if not isinstance(supported_params, list):
        return controls.supports_reasoning
    normalized = {str(param) for param in supported_params}
    supports_reasoning = (
        "reasoning" in normalized or "reasoning.effort" in normalized
    )
    return supports_reasoning or controls.supports_reasoning


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
        input_cost_per_1m=input_cost * scale
        if input_cost is not None
        else None,
        output_cost_per_1m=output_cost * scale
        if output_cost is not None
        else None,
        reasoning_cost_per_1m=(
            reasoning_cost * scale if reasoning_cost is not None else None
        ),
        raw=value,
    )


def _detect_supports_vision(item: dict[str, Any]) -> bool | None:
    modalities = item.get("input_modalities")
    if isinstance(modalities, list):
        return any(
            str(modality).lower() in {"image", "vision"}
            for modality in modalities
        )
    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        mods = architecture.get("input_modalities")
        if isinstance(mods, list):
            return any(
                str(modality).lower() in {"image", "vision"}
                for modality in mods
            )
    return None
