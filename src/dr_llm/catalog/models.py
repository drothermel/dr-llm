from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelCatalogPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str | None = "USD"
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
    supports_vision: bool | None = None
    pricing: ModelCatalogPricing | None = None
    rate_limits: ModelCatalogRateLimit | None = None
    source_quality: Literal["live", "static"] = "live"
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime | None = None


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
