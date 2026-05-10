from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from dr_llm.pool.db.names import (
    IndexNamePrefix,
    PoolIndexName,
    PoolTableType,
    SampleColumn,
    pool_index_name,
)
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables.table_def_protocol import (
    ColumnServerDefault,
    build_key_columns,
)


class SamplesTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.SAMPLES

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(SampleColumn.SAMPLE_ID, Text, primary_key=True),
            *build_key_columns(schema),
            Column(SampleColumn.SAMPLE_IDX, Integer, nullable=False),
            Column(SampleColumn.RUN_ID, Text),
            Column(
                SampleColumn.REQUEST_JSON,
                JSONB,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_JSONB),
            ),
            Column(SampleColumn.RESPONSE_JSON, JSONB(none_as_null=True)),
            Column(SampleColumn.FINISH_REASON, Text),
            Column(
                SampleColumn.ATTEMPT_COUNT,
                Integer,
                nullable=False,
                server_default=text(ColumnServerDefault.ZERO),
            ),
            Column(
                SampleColumn.METADATA_JSON,
                JSONB,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_JSONB),
            ),
            Column(
                SampleColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text(ColumnServerDefault.NOW),
            ),
        )

    def build_indexes(self, table: Table, schema: PoolSchema) -> list[Index]:
        key_columns = [table.c[name] for name in schema.key_column_names]
        return [
            Index(
                pool_index_name(IndexNamePrefix.UNIQUE, table.name, PoolIndexName.CELL),
                *key_columns,
                table.c[SampleColumn.SAMPLE_IDX],
                unique=True,
            ),
            Index(
                pool_index_name(
                    IndexNamePrefix.STANDARD, table.name, PoolIndexName.KEY
                ),
                *key_columns,
            ),
            Index(
                pool_index_name(
                    IndexNamePrefix.STANDARD,
                    table.name,
                    PoolIndexName.INCOMPLETE,
                ),
                table.c[SampleColumn.CREATED_AT],
                postgresql_where=table.c[SampleColumn.RESPONSE_JSON].is_(None),
            ),
        ]

    def select_columns(self, table: Table, schema: PoolSchema) -> list[Any]:
        return [
            table.c[SampleColumn.SAMPLE_ID],
            *(table.c[name] for name in schema.key_column_names),
            table.c[SampleColumn.SAMPLE_IDX],
            table.c[SampleColumn.RUN_ID],
            table.c[SampleColumn.REQUEST_JSON],
            table.c[SampleColumn.RESPONSE_JSON],
            table.c[SampleColumn.FINISH_REASON],
            table.c[SampleColumn.ATTEMPT_COUNT],
            table.c[SampleColumn.METADATA_JSON],
            table.c[SampleColumn.CREATED_AT],
        ]
