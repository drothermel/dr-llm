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

COUNTED_PENDING_STATUSES: frozenset[PendingStatus] = frozenset(
    {
        PendingStatus.pending,
        PendingStatus.leased,
        PendingStatus.failed,
    }
)
# Count/reporting APIs exclude terminal successes because they already contribute
# to the samples table totals.


class PendingStatusCounts(BaseModel):
    """Aggregate queue counts excluding terminal successes.

    ``promoted`` rows remain queryable as explicit lifecycle records in the
    pending table, but count/reporting APIs exclude them because they already
    contribute to samples-table totals.
    """

    model_config = ConfigDict(frozen=True)

    pending: int = 0
    leased: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.leased + self.failed

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
        by_status: dict[str, int] = {}
        for row in rows:
            raw_status = row["status"]
            status_key = (
                raw_status.value
                if isinstance(raw_status, PendingStatus)
                else str(raw_status)
            )
            by_status[status_key] = int(row["cnt"])
        return cls(
            **{
                status.value: by_status.get(status.value, 0)
                for status in COUNTED_PENDING_STATUSES
            }
        )
