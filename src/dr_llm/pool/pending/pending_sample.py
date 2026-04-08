from __future__ import annotations

import json
from datetime import datetime
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import key_values_from_row, parse_json_field
from dr_llm.pool.pending.pending_status import PendingStatus


class PendingSample(BaseModel):
    """Sample in pending state awaiting validation/promotion.

    Field declaration order matches DB column order. ``key_values`` is a
    placeholder whose slot is replaced at serialization time with the
    schema-defined key columns.
    """

    model_config = ConfigDict(frozen=True)

    pending_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int = 0
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
    priority: int = 0
    status: PendingStatus = PendingStatus.pending
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    attempt_count: int = 0
    created_at: datetime | None = None

    @field_validator("payload", "metadata", mode="before")
    @classmethod
    def _parse_json(cls, v: Any) -> Any:
        return parse_json_field(v) if isinstance(v, str) else v

    @field_serializer("payload", "metadata")
    def _dump_json(self, v: dict[str, Any]) -> str:
        return json.dumps(v, default=str)

    @field_serializer("status")
    def _dump_status(self, v: PendingStatus) -> str:
        return v.value

    @classmethod
    def db_select_columns(cls, schema: PoolSchema) -> list[str]:
        cols: list[str] = []
        for name, field in cls.model_fields.items():
            if name == "key_values":
                cols.extend(schema.key_column_names)
            else:
                cols.append(str(field.serialization_alias or name))
        return cols

    def to_db_insert_row(self, schema: PoolSchema) -> dict[str, Any]:
        missing = [kc for kc in schema.key_column_names if kc not in self.key_values]
        if missing:
            raise ValueError(
                "PendingSample.key_values must include all schema key columns "
                f"{list(schema.key_column_names)!r}; missing: {missing!r}"
            )
        dumped = self.model_dump(
            by_alias=True,
            exclude={"worker_id", "lease_expires_at", "attempt_count", "created_at"},
        )
        row: dict[str, Any] = {}
        for col_name, value in dumped.items():
            if col_name == "key_values":
                for kc in schema.key_column_names:
                    row[kc] = self.key_values[kc]
            else:
                row[col_name] = value
        return row

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> PendingSample:
        return cls.model_validate(
            {**row, "key_values": key_values_from_row(schema, dict(row))}
        )
