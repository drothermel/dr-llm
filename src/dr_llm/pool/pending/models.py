from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import key_values_from_row, parse_json_field

_CREATED_AT_COLUMN = "created_at"
_PENDING_SAMPLE_DB_INSERT_COLUMNS = (
    "pending_id",
    "sample_idx",
    "payload_json",
    "source_run_id",
    "metadata_json",
    "priority",
    "status",
)
_PENDING_SAMPLE_DB_SELECT_COLUMNS = _PENDING_SAMPLE_DB_INSERT_COLUMNS + (
    "worker_id",
    "lease_expires_at",
    "attempt_count",
    _CREATED_AT_COLUMN,
)


class PendingStatus(StrEnum):
    pending = "pending"
    leased = "leased"
    promoted = "promoted"
    failed = "failed"


class PendingSample(BaseModel):
    """Sample in pending state awaiting validation/promotion."""

    model_config = ConfigDict(frozen=True)

    _DB_INSERT_COLUMNS: ClassVar[tuple[str, ...]] = _PENDING_SAMPLE_DB_INSERT_COLUMNS
    _DB_SELECT_COLUMNS: ClassVar[tuple[str, ...]] = _PENDING_SAMPLE_DB_SELECT_COLUMNS

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

    @classmethod
    def db_select_columns(cls, schema: PoolSchema) -> list[str]:
        return ["pending_id", *schema.key_column_names, *cls._DB_SELECT_COLUMNS[1:]]

    def to_db_insert_row(self, schema: PoolSchema) -> dict[str, Any]:
        row: dict[str, Any] = {"pending_id": self.pending_id}
        for key_name in schema.key_column_names:
            row[key_name] = self.key_values[key_name]
        row["sample_idx"] = self.sample_idx
        row["payload_json"] = json.dumps(self.payload, default=str)
        row["source_run_id"] = self.source_run_id
        row["metadata_json"] = json.dumps(self.metadata, default=str)
        row["priority"] = self.priority
        row["status"] = self.status.value
        return row

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> PendingSample:
        return cls(
            pending_id=str(row["pending_id"]),
            key_values=key_values_from_row(schema, dict(row)),
            sample_idx=int(row["sample_idx"]),
            payload=parse_json_field(row["payload_json"]),
            source_run_id=row["source_run_id"],
            metadata=parse_json_field(row["metadata_json"]),
            priority=int(row["priority"]),
            status=PendingStatus(row["status"]),
            worker_id=row["worker_id"],
            lease_expires_at=row["lease_expires_at"],
            attempt_count=int(row["attempt_count"]),
            created_at=row["created_at"],
        )


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
