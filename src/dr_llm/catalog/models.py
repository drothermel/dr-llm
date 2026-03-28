from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelCatalogSyncResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    success: bool
    entry_count: int = 0
    snapshot_id: str | None = None
    error: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
