"""Pool completion progress snapshot."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PoolProgress(BaseModel):
    """Snapshot of pool completion state.

    Invariants: total == incomplete + complete, leased <= incomplete, error <= complete.
    """

    model_config = ConfigDict(frozen=True)

    total: int
    incomplete: int
    leased: int
    complete: int
    error: int
