from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dr_llm.pool.pool_sample import PoolSample


class InsertResult(BaseModel):
    """Result of a bulk insert operation."""

    model_config = ConfigDict(frozen=True)

    inserted: int = 0
    skipped: int = 0
    failed: int = 0

    def __add__(self, other: InsertResult) -> InsertResult:
        return InsertResult(
            inserted=self.inserted + other.inserted,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
        )


class AcquireQuery(BaseModel):
    """Query for no-replacement sample acquisition."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    n: int = Field(ge=0)
    consumer_tag: str = ""


class AcquireResult(BaseModel):
    """Result of a sample acquisition."""

    model_config = ConfigDict(frozen=True)

    samples: list[PoolSample] = Field(default_factory=list)

    @computed_field
    @property
    def claimed(self) -> int:
        return len(self.samples)

    def deficit(self, requested_n: int) -> int:
        return max(requested_n - len(self.samples), 0)


class PoolClaim(BaseModel):
    """Claim record for no-replacement tracking."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    request_id: str
    consumer_tag: str = ""
    sample_id: str
    claim_idx: int
    claimed_at: datetime | None = None


class CoverageRow(BaseModel):
    """Aggregate count per unique key combination."""

    model_config = ConfigDict(frozen=True)

    key_values: dict[str, Any] = Field(default_factory=dict)
    count: int = 0
