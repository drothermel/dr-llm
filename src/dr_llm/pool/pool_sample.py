from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

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

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int | None = None
    payload: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("payload", "payload_json"),
        serialization_alias="payload_json",
    )
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "metadata_json"),
        serialization_alias="metadata_json",
    )
    status: SampleStatus = SampleStatus.active
    created_at: datetime | None = None

    @field_validator("payload", "metadata", mode="before")
    @classmethod
    def _parse_json(cls, v: Any) -> Any:
        return parse_json_field(v) if isinstance(v, str) else v

    def to_db_insert_row(self, schema: PoolSchema) -> dict[str, Any]:
        missing = set(schema.key_column_names) - set(self.key_values.keys())
        if missing:
            raise ValueError(
                f"Missing key columns for PoolSample: {missing}. "
                f"Expected: {set(schema.key_column_names)}"
            )
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
