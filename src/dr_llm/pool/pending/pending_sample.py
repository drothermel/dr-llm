from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from dr_llm.pool.keyed_sample_base import KeyedSampleBase
from dr_llm.pool.pending.pending_status import PendingStatus


class PendingSample(KeyedSampleBase):
    """Sample in pending state awaiting validation/promotion."""

    pending_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_idx: int = 0
    priority: int = 0
    status: PendingStatus = PendingStatus.pending
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    attempt_count: int = 0
    created_at: datetime | None = None

    def to_db_insert_row(self) -> dict[str, Any]:
        return {
            "pending_id": self.pending_id,
            "sample_idx": self.sample_idx,
            "priority": self.priority,
            "status": self.status.value,
            **self._base_insert_row(),
        }
