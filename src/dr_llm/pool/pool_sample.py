from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field


class PoolSample(BaseModel):
    """A single pool sample row."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int | None = None
    run_id: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | None = None
    finish_reason: str | None = None
    attempt_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None

    @computed_field
    @property
    def is_complete(self) -> bool:
        return self.response is not None
