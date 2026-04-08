from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm.providers.reasoning_capability_types import ReasoningCapabilities


def _derive_supports_reasoning_from_capabilities(
    capabilities: Any,
) -> bool | None:
    if capabilities is None:
        return None
    if isinstance(capabilities, ReasoningCapabilities):
        return capabilities.supports_reasoning
    if isinstance(capabilities, dict):
        mode = capabilities.get("mode", "unsupported")
        return mode != "unsupported"
    return None


class ModelCatalogPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str = "USD"
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    reasoning_cost_per_1m: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogRateLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    concurrency_limit: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_reasoning: bool | None = None
    reasoning_capabilities: ReasoningCapabilities | None = None
    supports_vision: bool | None = None
    pricing: ModelCatalogPricing | None = None
    rate_limits: ModelCatalogRateLimit | None = None
    source_quality: Literal["live", "static"] = "live"
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _populate_supports_reasoning(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        derived = _derive_supports_reasoning_from_capabilities(
            data.get("reasoning_capabilities")
        )
        if derived is not None:
            data["supports_reasoning"] = derived
        return data


class ModelCatalogQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str | None = None
    supports_reasoning: bool | None = None
    model_contains: str | None = None
    limit: int = 200
    offset: int = 0


class ModelCatalogSyncResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    success: bool
    entry_count: int = 0
    snapshot_id: str | None = None
    error: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
