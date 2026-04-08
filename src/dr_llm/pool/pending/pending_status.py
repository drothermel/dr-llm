from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

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

    @property
    def in_flight(self) -> int:
        """Samples not yet in a terminal state (pending + leased)."""
        return self.pending + self.leased

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        status_key: str = "status",
        count_key: str = "cnt",
    ) -> PendingStatusCounts:
        """Build counts from grouped query rows, defaulting unknown statuses to 0."""
        counts: dict[str, int] = {s.value: 0 for s in PendingStatus}
        for row in rows:
            status = row[status_key]
            if status in counts:
                counts[status] = int(row[count_key])
        return cls(**counts)
