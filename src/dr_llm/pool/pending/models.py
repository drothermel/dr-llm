from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PendingStatus(StrEnum):
    pending = "pending"
    leased = "leased"
    promoted = "promoted"
    failed = "failed"


class PendingSample(BaseModel):
    """Sample in pending state awaiting validation/promotion."""

    model_config = ConfigDict(frozen=True)

    pending_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    status: PendingStatus = PendingStatus.pending
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    attempt_count: int = 0
    created_at: datetime | None = None


class PendingStatusCounts(BaseModel):
    """Counts of pending rows by lifecycle status."""

    model_config = ConfigDict(frozen=True)

    pending: int = 0
    leased: int = 0
    promoted: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.leased + self.promoted + self.failed
