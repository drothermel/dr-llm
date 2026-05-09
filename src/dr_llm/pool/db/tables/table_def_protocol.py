from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Protocol

from sqlalchemy import Boolean, Column, Double, Index, Integer, MetaData, Table, Text

from dr_llm.pool.db.names import PoolTableType
from dr_llm.pool.db.schema import ColumnType, PoolSchema


_TYPE_MAP: dict[ColumnType, type[Any]] = {
    ColumnType.text: Text,
    ColumnType.integer: Integer,
    ColumnType.boolean: Boolean,
    ColumnType.float_: Double,
}


class ColumnServerDefault(StrEnum):
    EMPTY_JSONB = "'{}'::jsonb"
    EMPTY_TEXT = "''"
    NOW = "now()"
    ONE = "1"
    PENDING_STATUS = "'pending'"
    ZERO = "0"


class TableDef(Protocol):
    table_type: ClassVar[PoolTableType]

    def build_table(self, schema: PoolSchema, metadata: MetaData, /) -> Table: ...

    def build_indexes(self, table: Table, schema: PoolSchema, /) -> list[Index]: ...

    def select_columns(self, table: Table, schema: PoolSchema, /) -> list[Any]: ...


def build_key_columns(schema: PoolSchema) -> list[Column[Any]]:
    return [
        Column(key.name, _TYPE_MAP[key.type], nullable=False)
        for key in schema.key_columns
    ]
