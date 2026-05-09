from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Double, Index, Integer, MetaData, Table, Text, text
from sqlalchemy.dialects.postgresql import TIMESTAMP

from dr_llm.pool.db.names import CallStatsColumn, PoolTableType
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables.table_def_protocol import ColumnServerDefault


class CallStatsTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[PoolTableType] = PoolTableType.CALL_STATS

    def build_table(self, schema: PoolSchema, metadata: MetaData) -> Table:
        return Table(
            schema.table_name(self.table_type),
            metadata,
            Column(CallStatsColumn.SAMPLE_ID, Text, primary_key=True),
            Column(CallStatsColumn.LATENCY_MS, Integer, nullable=False),
            Column(CallStatsColumn.TOTAL_COST_USD, Double),
            Column(CallStatsColumn.PROMPT_TOKENS, Integer, nullable=False),
            Column(CallStatsColumn.COMPLETION_TOKENS, Integer, nullable=False),
            Column(CallStatsColumn.REASONING_TOKENS, Integer),
            Column(CallStatsColumn.TOTAL_TOKENS, Integer, nullable=False),
            Column(
                CallStatsColumn.ATTEMPT_COUNT,
                Integer,
                nullable=False,
                server_default=text(ColumnServerDefault.ONE),
            ),
            Column(CallStatsColumn.FINISH_REASON, Text),
            Column(
                CallStatsColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text(ColumnServerDefault.NOW),
            ),
        )

    def build_indexes(self, table: Table, schema: PoolSchema) -> list[Index]:
        return []

    def select_columns(self, table: Table, schema: PoolSchema) -> list[Any]:
        return list(table.c)
