from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PendingStatus(StrEnum):
    pending = "pending"
    leased = "leased"
    promoted = "promoted"
    failed = "failed"


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
