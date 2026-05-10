from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text
from sqlalchemy.dialects.postgresql import TIMESTAMP

from dr_llm.pool.db.tables.table_def_protocol import ColumnServerDefault
from dr_llm.sampling.db.names import (
    ClaimColumn,
    ClaimsIndexName,
    ClaimsTableType,
    IndexNamePrefix,
    claims_index_name,
)


class ClaimsTableDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_type: ClassVar[ClaimsTableType] = ClaimsTableType.CLAIMS

    def build_table(self, table_name: str, metadata: MetaData) -> Table:
        return Table(
            table_name,
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

    def build_indexes(self, table: Table) -> list[Index]:
        return [
            Index(
                claims_index_name(
                    IndexNamePrefix.UNIQUE, table.name, ClaimsIndexName.RUN_SAMPLE
                ),
                table.c[ClaimColumn.RUN_ID],
                table.c[ClaimColumn.SAMPLE_ID],
                unique=True,
            ),
            Index(
                claims_index_name(
                    IndexNamePrefix.STANDARD, table.name, ClaimsIndexName.RUN
                ),
                table.c[ClaimColumn.RUN_ID],
            ),
        ]

    def select_columns(self, table: Table) -> list[Any]:
        return list(table.c)
