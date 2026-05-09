from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from dr_llm.pool.db.names import (
    IndexNamePrefix,
    PendingColumn,
    PoolIndexName,
    PoolTableType,
    pool_index_name,
)
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables.table_def_protocol import (
    ColumnServerDefault,
    build_key_columns,
)


class PendingTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.PENDING

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(PendingColumn.PENDING_ID, Text, primary_key=True),
            *build_key_columns(schema),
            Column(PendingColumn.SAMPLE_IDX, Integer, nullable=False),
            Column(
                PendingColumn.PAYLOAD_JSON,
                JSONB,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_JSONB),
            ),
            Column(PendingColumn.SOURCE_RUN_ID, Text),
            Column(
                PendingColumn.METADATA_JSON,
                JSONB,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_JSONB),
            ),
            Column(
                PendingColumn.PRIORITY,
                Integer,
                nullable=False,
                server_default=text(ColumnServerDefault.ZERO),
            ),
            Column(
                PendingColumn.STATUS,
                Text,
                nullable=False,
                server_default=text(ColumnServerDefault.PENDING_STATUS),
            ),
            Column(PendingColumn.WORKER_ID, Text),
            Column(PendingColumn.LEASE_EXPIRES_AT, TIMESTAMP(timezone=True)),
            Column(
                PendingColumn.ATTEMPT_COUNT,
                Integer,
                nullable=False,
                server_default=text(ColumnServerDefault.ZERO),
            ),
            Column(
                PendingColumn.CREATED_AT,
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
                table.c[PendingColumn.SAMPLE_IDX],
                unique=True,
            ),
            Index(
                pool_index_name(
                    IndexNamePrefix.STANDARD,
                    table.name,
                    PoolIndexName.STATUS_PRIORITY,
                ),
                table.c[PendingColumn.STATUS],
                table.c[PendingColumn.PRIORITY].desc(),
                table.c[PendingColumn.CREATED_AT].asc(),
            ),
            Index(
                pool_index_name(
                    IndexNamePrefix.STANDARD, table.name, PoolIndexName.KEY
                ),
                *key_columns,
            ),
        ]

    def select_columns(self, table: Table, schema: PoolSchema) -> list[Any]:
        return [
            table.c[PendingColumn.PENDING_ID],
            *(table.c[name] for name in schema.key_column_names),
            table.c[PendingColumn.SAMPLE_IDX],
            table.c[PendingColumn.PAYLOAD_JSON],
            table.c[PendingColumn.SOURCE_RUN_ID],
            table.c[PendingColumn.METADATA_JSON],
            table.c[PendingColumn.PRIORITY],
            table.c[PendingColumn.STATUS],
            table.c[PendingColumn.WORKER_ID],
            table.c[PendingColumn.LEASE_EXPIRES_AT],
            table.c[PendingColumn.ATTEMPT_COUNT],
            table.c[PendingColumn.CREATED_AT],
        ]
