"""Pool completion progress snapshot."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


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

    @model_validator(mode="after")
    def _validate_invariants(self) -> PoolProgress:
        if self.total != self.incomplete + self.complete:
            raise ValueError(
                "PoolProgress.total must equal incomplete + complete "
                f"(total={self.total}, incomplete={self.incomplete}, "
                f"complete={self.complete})"
            )
        if self.leased > self.incomplete:
            raise ValueError(
                "PoolProgress.leased must be <= incomplete "
                f"(leased={self.leased}, incomplete={self.incomplete})"
            )
        if self.error > self.complete:
            raise ValueError(
                "PoolProgress.error must be <= complete "
                f"(error={self.error}, complete={self.complete})"
            )
        return self
