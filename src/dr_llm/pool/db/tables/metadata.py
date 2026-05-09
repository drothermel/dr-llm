from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, MetaData, PrimaryKeyConstraint, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from dr_llm.pool.db.names import MetadataColumn, PoolTableType
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables.table_def_protocol import ColumnServerDefault


class MetadataTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.METADATA

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(MetadataColumn.POOL_NAME, Text, nullable=False),
            Column(MetadataColumn.KEY, Text, nullable=False),
            Column(MetadataColumn.VALUE_JSON, JSONB, nullable=False),
            Column(
                MetadataColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text(ColumnServerDefault.NOW),
            ),
            Column(
                MetadataColumn.UPDATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text(ColumnServerDefault.NOW),
            ),
            PrimaryKeyConstraint(MetadataColumn.POOL_NAME, MetadataColumn.KEY),
        )

    def build_indexes(self, table: Table, schema: PoolSchema) -> list[Index]:
        return []

    def select_columns(self, table: Table, schema: PoolSchema) -> list[Any]:
        return list(table.c)
