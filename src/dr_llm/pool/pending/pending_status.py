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


IN_FLIGHT_STATUSES: frozenset[PendingStatus] = frozenset(
    {PendingStatus.pending, PendingStatus.leased}
)
# Non-terminal pending statuses: rows still waiting to be processed or currently
# leased by a worker.


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
    ) -> PendingStatusCounts:
        """Build counts from grouped query rows, defaulting unknown statuses to 0."""
        by_status = {row["status"]: int(row["cnt"]) for row in rows}
        return cls(**{s: by_status.get(s, 0) for s in PendingStatus})
