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
_POOL_SAMPLE_DB_INSERT_COLUMNS = (
    "sample_id",
    "sample_idx",
    "payload_json",
    "source_run_id",
    "metadata_json",
    "status",
)
_POOL_SAMPLE_DB_SELECT_COLUMNS = _POOL_SAMPLE_DB_INSERT_COLUMNS + (_CREATED_AT_COLUMN,)


class SampleStatus(StrEnum):
    active = "active"
    superseded = "superseded"


class PoolSample(BaseModel):
    """A single pool sample row."""

    model_config = ConfigDict(frozen=True)

    _DB_INSERT_COLUMNS: ClassVar[tuple[str, ...]] = _POOL_SAMPLE_DB_INSERT_COLUMNS
    _DB_SELECT_COLUMNS: ClassVar[tuple[str, ...]] = _POOL_SAMPLE_DB_SELECT_COLUMNS

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_idx: int | None = None
    key_values: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: SampleStatus = SampleStatus.active
    created_at: datetime | None = None

    @classmethod
    def db_select_columns(cls, schema: PoolSchema) -> list[str]:
        return ["sample_id", *schema.key_column_names, *cls._DB_SELECT_COLUMNS[1:]]

    def to_db_insert_row(self, schema: PoolSchema) -> dict[str, Any]:
        row: dict[str, Any] = {"sample_id": self.sample_id}
        for key_name in schema.key_column_names:
            row[key_name] = self.key_values[key_name]
        row["sample_idx"] = self.sample_idx
        row["payload_json"] = json.dumps(self.payload, default=str)
        row["source_run_id"] = self.source_run_id
        row["metadata_json"] = json.dumps(self.metadata, default=str)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> PoolSample:
        return cls(
            sample_id=str(row["sample_id"]),
            sample_idx=row["sample_idx"],
            key_values=key_values_from_row(schema, dict(row)),
            payload=parse_json_field(row["payload_json"]),
            source_run_id=row["source_run_id"],
            metadata=parse_json_field(row["metadata_json"]),
            status=SampleStatus(row["status"]),
            created_at=row["created_at"],
        )


class AcquireQuery(BaseModel):
    """Query for no-replacement sample acquisition."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    n: int
    consumer_tag: str = ""


class AcquireResult(BaseModel):
    """Result of a sample acquisition."""

    model_config = ConfigDict(frozen=True)

    samples: list[PoolSample] = Field(default_factory=list)
    claimed: int = 0

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
