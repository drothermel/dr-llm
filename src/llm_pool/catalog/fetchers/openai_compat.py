from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_pool.catalog.fetchers.common import api_key_from_env, get_json
from llm_pool.providers.openai_compat import OpenAICompatAdapter
from llm_pool.types import ModelCatalogEntry, ModelCatalogPricing


def fetch_openai_compat_models(
    adapter: OpenAICompatAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    base = adapter._config.base_url.rstrip("/")  # noqa: SLF001
    endpoint = f"{base}/models"
    key = adapter._config.api_key or api_key_from_env(adapter._config.api_key_env)  # noqa: SLF001
    headers: dict[str, str] | None = None
    if key:
        headers = {"Authorization": f"Bearer {key}"}
    payload = get_json(url=endpoint, headers=headers)
    items_raw = payload.get("data")
    items = items_raw if isinstance(items_raw, list) else []
    now = datetime.now(timezone.utc)
    out: list[ModelCatalogEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or item.get("name") or "").strip()
        if not model_id:
            continue
        pricing = _parse_pricing(item.get("pricing"))
        supported_params = item.get("supported_parameters")
        supports_reasoning = None
        supports_tools = None
        if isinstance(supported_params, list):
            normalized = {str(param) for param in supported_params}
            supports_reasoning = (
                "reasoning" in normalized or "reasoning.effort" in normalized
            )
            supports_tools = (
                "tools" in normalized
                or "tool_choice" in normalized
                or "function_call" in normalized
            )
        out.append(
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_id,
                display_name=str(item.get("name") or model_id),
                context_window=_as_int(item.get("context_length")),
                max_output_tokens=_as_int(item.get("max_output_tokens")),
                supports_reasoning=supports_reasoning,
                supports_tools=supports_tools,
                supports_vision=_detect_supports_vision(item),
                pricing=pricing,
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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _parse_pricing(value: Any) -> ModelCatalogPricing | None:
    if not isinstance(value, dict):
        return None
    input_cost = (
        _as_float(value.get("prompt"))
        if value.get("prompt") is not None
        else _as_float(value.get("input"))
    )
    output_cost = (
        _as_float(value.get("completion"))
        if value.get("completion") is not None
        else _as_float(value.get("output"))
    )
    reasoning_cost = _as_float(value.get("reasoning"))
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
