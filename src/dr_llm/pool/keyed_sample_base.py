"""Shared base for sample models with key_values + JSON payload/metadata.

Both ``PoolSample`` and ``PendingSample`` serialize ``key_values`` into
schema-defined key columns and persist ``payload``/``metadata`` as JSONB.
This module centralizes that contract so the two sample types stay in sync.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import key_values_from_row


def json_dict_field(name: str) -> Any:
    """Build a Field for a dict that round-trips as ``{name}_json`` in the DB."""
    json_alias = f"{name}_json"
    return Field(
        default_factory=dict,
        validation_alias=AliasChoices(name, json_alias),
        serialization_alias=json_alias,
    )


class KeyedSampleBase(BaseModel):
    """Base for pool sample models that serialize to/from DB rows.

    Subclasses inherit ``key_values``, ``payload``, ``metadata``, and
    ``source_run_id``, and must implement ``to_db_insert_row`` to assemble
    their per-row dict directly from instance attributes (see the rationale
    on that method).
    """

    model_config = ConfigDict(frozen=True)

    key_values: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = json_dict_field("payload")
    metadata: dict[str, Any] = json_dict_field("metadata")
    source_run_id: str | None = None

    @classmethod
    def from_db_row(cls, schema: PoolSchema, row: Mapping[str, Any]) -> Self:
        row_dict = dict(row)
        return cls(**row_dict, key_values=key_values_from_row(schema, row_dict))

    def _base_insert_row(self) -> dict[str, Any]:
        """Shared payload/metadata/source_run_id + key_values for insert rows.

        Subclasses splat this in ``to_db_insert_row`` and add their own
        type-specific columns. Bypasses ``model_dump`` so batch inserts of
        hundreds of samples don't run the full Pydantic serialization pipeline
        per row. Only the JSONB payload/metadata dicts go through
        ``to_jsonable_python`` to coerce nested ``BaseModel`` / ``datetime``
        values into JSON-safe shapes.
        """
        return {
            "payload_json": to_jsonable_python(self.payload),
            "metadata_json": to_jsonable_python(self.metadata),
            "source_run_id": self.source_run_id,
            **self.key_values,
        }

    @abstractmethod
    def to_db_insert_row(self) -> dict[str, Any]:
        """Build the DB insert row directly from instance attributes."""
