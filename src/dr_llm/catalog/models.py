from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.types import ModelCatalogEntry


DEFAULT_MODEL_OVERRIDES_PATH = Path("config/model_overrides.json")


class ModelCatalogSyncResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    success: bool
    entry_count: int = 0
    snapshot_id: str | None = None
    error: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class ModelOverridesFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: list[ModelCatalogEntry] = Field(default_factory=list)
