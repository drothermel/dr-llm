from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field
from pydantic_core import to_jsonable_python

from dr_llm.pool.db import PoolSchema, SampleColumn
from dr_llm.pool.db.sql_helpers import key_values_from_row


def json_dict_field(name: str) -> Any:
    json_alias = f"{name}_json"
    return Field(
        default_factory=dict,
        validation_alias=AliasChoices(name, json_alias),
        serialization_alias=json_alias,
    )


def optional_json_dict_field(name: str) -> Any:
    json_alias = f"{name}_json"
    return Field(
        default=None,
        validation_alias=AliasChoices(name, json_alias),
        serialization_alias=json_alias,
    )


class PoolSample(BaseModel):
    """A single pool sample row."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int | None = None
    run_id: str | None = None
    request: dict[str, Any] = json_dict_field("request")
    response: dict[str, Any] | None = optional_json_dict_field("response")
    finish_reason: str | None = None
    attempt_count: int = 0
    metadata: dict[str, Any] = json_dict_field("metadata")
    created_at: datetime | None = None

    @computed_field
    @property
    def is_complete(self) -> bool:
        return self.response is not None

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> Self:
        row_dict = dict(row)
        return cls(**row_dict, key_values=key_values_from_row(schema, row_dict))

    def to_db_insert_row(self) -> dict[str, Any]:
        return {
            SampleColumn.SAMPLE_ID: self.sample_id,
            SampleColumn.SAMPLE_IDX: self.sample_idx,
            SampleColumn.RUN_ID: self.run_id,
            SampleColumn.REQUEST_JSON: to_jsonable_python(self.request),
            SampleColumn.RESPONSE_JSON: to_jsonable_python(self.response),
            SampleColumn.FINISH_REASON: self.finish_reason,
            SampleColumn.ATTEMPT_COUNT: self.attempt_count,
            SampleColumn.METADATA_JSON: to_jsonable_python(self.metadata),
            **self.key_values,
        }
