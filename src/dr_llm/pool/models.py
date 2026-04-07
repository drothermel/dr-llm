from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import key_values_from_row, parse_json_field


class SampleStatus(StrEnum):
    active = "active"
    superseded = "superseded"


class PoolSample(BaseModel):
    """A single pool sample row.

    Field declaration order matches DB column order. ``key_values`` is a
    placeholder whose slot is replaced at serialization time with the
    schema-defined key columns.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict, alias="payload_json")
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_json")
    status: SampleStatus = SampleStatus.active
    created_at: datetime | None = None

    @field_validator("payload", "metadata", mode="before")
    @classmethod
    def _parse_json(cls, v: Any) -> Any:
        return parse_json_field(v) if isinstance(v, str) else v

    @field_serializer("payload", "metadata")
    def _dump_json(self, v: dict[str, Any]) -> str:
        return json.dumps(v, default=str)

    @field_serializer("status")
    def _dump_status(self, v: SampleStatus) -> str:
        return v.value

    @classmethod
    def db_select_columns(cls, schema: PoolSchema) -> list[str]:
        cols: list[str] = []
        for name, field in cls.model_fields.items():
            if name == "key_values":
                cols.extend(schema.key_column_names)
            else:
                cols.append(field.alias or name)
        return cols

    def to_db_insert_row(self, schema: PoolSchema) -> dict[str, Any]:
        dumped = self.model_dump(by_alias=True, exclude={"created_at"})
        row: dict[str, Any] = {}
        for col_name, value in dumped.items():
            if col_name == "key_values":
                for kc in schema.key_column_names:
                    row[kc] = self.key_values[kc]
            else:
                row[col_name] = value
        return row

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> PoolSample:
        return cls.model_validate(
            {**row, "key_values": key_values_from_row(schema, dict(row))}
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
