from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP

from dr_llm.pool.db.names import LeaseColumn, PoolTableType
from dr_llm.pool.db.schema import PoolSchema


class LeasesTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.LEASES

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(LeaseColumn.SAMPLE_ID, Text, primary_key=True),
            Column(LeaseColumn.WORKER_ID, Text, nullable=False),
            Column(
                LeaseColumn.LEASE_EXPIRES_AT, TIMESTAMP(timezone=True), nullable=False
            ),
        )

    def build_indexes(self, _table: Table, _schema: PoolSchema) -> list[Index]:
        return []

    def select_columns(self, table: Table, _schema: PoolSchema) -> list[Any]:
        return list(table.c)
