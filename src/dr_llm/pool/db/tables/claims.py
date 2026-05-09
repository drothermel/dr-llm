from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text
from sqlalchemy.dialects.postgresql import TIMESTAMP

from dr_llm.pool.db.names import (
    ClaimColumn,
    IndexNamePrefix,
    PoolIndexName,
    PoolTableType,
    pool_index_name,
)
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables.table_def_protocol import ColumnServerDefault


class ClaimsTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.CLAIMS

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(ClaimColumn.CLAIM_ID, Text, primary_key=True),
            Column(ClaimColumn.RUN_ID, Text, nullable=False),
            Column(ClaimColumn.REQUEST_ID, Text, nullable=False),
            Column(
                ClaimColumn.CONSUMER_TAG,
                Text,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_TEXT),
            ),
            Column(ClaimColumn.SAMPLE_ID, Text, nullable=False),
            Column(ClaimColumn.CLAIM_IDX, Integer, nullable=False),
            Column(
                ClaimColumn.CLAIMED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text(ColumnServerDefault.NOW),
            ),
        )

    def build_indexes(self, table: Table, schema: PoolSchema) -> list[Index]:
        return [
            Index(
                pool_index_name(
                    IndexNamePrefix.UNIQUE, table.name, PoolIndexName.RUN_SAMPLE
                ),
                table.c[ClaimColumn.RUN_ID],
                table.c[ClaimColumn.SAMPLE_ID],
                unique=True,
            ),
            Index(
                pool_index_name(
                    IndexNamePrefix.STANDARD, table.name, PoolIndexName.RUN
                ),
                table.c[ClaimColumn.RUN_ID],
            ),
        ]

    def select_columns(self, table: Table, schema: PoolSchema) -> list[Any]:
        return list(table.c)
