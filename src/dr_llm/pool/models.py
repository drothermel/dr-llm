from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.project.project_info import ProjectInfo

_POOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


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


class PoolInspectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str

    @field_validator("project_name", "pool_name")
    @classmethod
    def _normalize_names(cls, value: str) -> str:
        return value.strip()


class CreatePoolRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    key_axes: list[str] = Field(default_factory=list)

    @field_validator("project_name", "pool_name")
    @classmethod
    def _normalize_names(cls, value: str) -> str:
        return value.strip()

    @field_validator("key_axes")
    @classmethod
    def _normalize_key_axes(cls, value: list[str]) -> list[str]:
        return [axis.strip() for axis in value if axis.strip()]

    @classmethod
    def from_csv(
        cls, *, project_name: str, pool_name: str, axes_csv: str
    ) -> CreatePoolRequest:
        return cls(
            project_name=project_name,
            pool_name=pool_name,
            key_axes=[axis.strip() for axis in axes_csv.split(",") if axis.strip()],
        )

    @computed_field
    @property
    def has_key_axes(self) -> bool:
        return bool(self.key_axes)

    @computed_field
    @property
    def pool_name_is_valid(self) -> bool:
        return bool(_POOL_NAME_RE.match(self.pool_name))


class PoolInspectionStatus(StrEnum):
    empty = "empty"
    in_progress = "in_progress"
    complete = "complete"


class PoolInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    name: str
    pool_schema: PoolSchema
    created_at: datetime | None = None
    sample_count: int = 0
    pending_counts: PendingStatusCounts = Field(default_factory=PendingStatusCounts)
    status: PoolInspectionStatus

    @computed_field
    @property
    def pending_total(self) -> int:
        return self.pending_counts.total

    @computed_field
    @property
    def in_flight(self) -> int:
        return self.pending_counts.in_flight


class PoolCreationBlockReason(StrEnum):
    invalid_pool_name = "invalid_pool_name"
    missing_key_axes = "missing_key_axes"
    invalid_key_axis = "invalid_key_axis"
    project_not_found = "project_not_found"
    project_not_running = "project_not_running"
    pool_already_exists = "pool_already_exists"
    max_pools_reached = "max_pools_reached"
    pool_in_progress = "pool_in_progress"
    cooldown_active = "cooldown_active"


class PoolCreationViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: PoolCreationBlockReason
    message: str
    project_name: str | None = None
    pool_name: str | None = None


class PoolCreationReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: CreatePoolRequest
    project: ProjectInfo | None = None
    existing_pools: list[PoolInspection] = Field(default_factory=list)
    violations: list[PoolCreationViolation] = Field(default_factory=list)

    @computed_field
    @property
    def allowed(self) -> bool:
        return not self.violations
