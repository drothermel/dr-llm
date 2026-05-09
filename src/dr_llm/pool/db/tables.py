from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Double,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from dr_llm.pool.db.names import (
    CallStatsColumn,
    ClaimColumn,
    IndexNamePrefix,
    MetadataColumn,
    PendingColumn,
    PoolIndexName,
    PoolTableType,
    SampleColumn,
    pool_index_name,
)
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


class PoolTables:
    def __init__(self, schema: PoolSchema) -> None:
        self.schema = schema
        self.sa_metadata = MetaData()
        self.tables: dict[PoolTableType, Table] = {
            PoolTableType.SAMPLES: self._build_samples_table(),
            PoolTableType.CLAIMS: self._build_claims_table(),
            PoolTableType.PENDING: self._build_pending_table(),
            PoolTableType.METADATA: self._build_metadata_table(),
            PoolTableType.CALL_STATS: self._build_call_stats_table(),
        }
        self._build_indexes()

    def __getitem__(self, table_type: PoolTableType) -> Table:
        return self.tables[table_type]

    @property
    def all_tables(self) -> list[Table]:
        return [self.tables[table_type] for table_type in PoolTableType]

    def key_columns(self, table_type: PoolTableType) -> list[Column[Any]]:
        if table_type not in {PoolTableType.SAMPLES, PoolTableType.PENDING}:
            msg = f"{table_type} does not have pool key columns"
            raise ValueError(msg)
        table = self[table_type]
        return [table.c[name] for name in self.schema.key_column_names]

    def select_columns(self, table_type: PoolTableType) -> list[Any]:
        if table_type == PoolTableType.SAMPLES:
            samples = self[PoolTableType.SAMPLES]
            return [
                samples.c[SampleColumn.SAMPLE_ID],
                *self.key_columns(PoolTableType.SAMPLES),
                samples.c[SampleColumn.SAMPLE_IDX],
                samples.c[SampleColumn.PAYLOAD_JSON],
                samples.c[SampleColumn.SOURCE_RUN_ID],
                samples.c[SampleColumn.METADATA_JSON],
                samples.c[SampleColumn.CREATED_AT],
            ]
        if table_type == PoolTableType.PENDING:
            pending = self[PoolTableType.PENDING]
            return [
                pending.c[PendingColumn.PENDING_ID],
                *self.key_columns(PoolTableType.PENDING),
                pending.c[PendingColumn.SAMPLE_IDX],
                pending.c[PendingColumn.PAYLOAD_JSON],
                pending.c[PendingColumn.SOURCE_RUN_ID],
                pending.c[PendingColumn.METADATA_JSON],
                pending.c[PendingColumn.PRIORITY],
                pending.c[PendingColumn.STATUS],
                pending.c[PendingColumn.WORKER_ID],
                pending.c[PendingColumn.LEASE_EXPIRES_AT],
                pending.c[PendingColumn.ATTEMPT_COUNT],
                pending.c[PendingColumn.CREATED_AT],
            ]
        msg = f"{table_type} does not have a select-column projection"
        raise ValueError(msg)

    def ensure_indexes(self, bind: Connection) -> None:
        """Backfill any missing named indexes for runtime-owned pool tables."""
        for table in self.all_tables:
            for index in table.indexes:
                index.create(bind=bind, checkfirst=True)

    def _build_samples_table(self) -> Table:
        return Table(
            self.schema.table_name(PoolTableType.SAMPLES),
            self.sa_metadata,
            Column(SampleColumn.SAMPLE_ID, Text, primary_key=True),
            *self._key_columns(),
            Column(SampleColumn.SAMPLE_IDX, Integer, nullable=False),
            Column(
                SampleColumn.PAYLOAD_JSON,
                JSONB,
                nullable=False,
                server_default=text(ColumnServerDefault.EMPTY_JSONB),
            ),
            Column(SampleColumn.SOURCE_RUN_ID, Text),
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

    def _build_claims_table(self) -> Table:
        return Table(
            self.schema.table_name(PoolTableType.CLAIMS),
            self.sa_metadata,
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

    def _build_pending_table(self) -> Table:
        return Table(
            self.schema.table_name(PoolTableType.PENDING),
            self.sa_metadata,
            Column(PendingColumn.PENDING_ID, Text, primary_key=True),
            *self._key_columns(),
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

    def _build_indexes(self) -> None:
        samples = self[PoolTableType.SAMPLES]
        claims = self[PoolTableType.CLAIMS]
        pending = self[PoolTableType.PENDING]
        samples_key_columns = self.key_columns(PoolTableType.SAMPLES)
        pending_key_columns = self.key_columns(PoolTableType.PENDING)

        Index(
            pool_index_name(IndexNamePrefix.UNIQUE, samples.name, PoolIndexName.CELL),
            *samples_key_columns,
            samples.c[SampleColumn.SAMPLE_IDX],
            unique=True,
        )
        Index(
            pool_index_name(IndexNamePrefix.STANDARD, samples.name, PoolIndexName.KEY),
            *samples_key_columns,
        )
        Index(
            pool_index_name(
                IndexNamePrefix.UNIQUE, claims.name, PoolIndexName.RUN_SAMPLE
            ),
            claims.c[ClaimColumn.RUN_ID],
            claims.c[ClaimColumn.SAMPLE_ID],
            unique=True,
        )
        Index(
            pool_index_name(IndexNamePrefix.STANDARD, claims.name, PoolIndexName.RUN),
            claims.c[ClaimColumn.RUN_ID],
        )
        Index(
            pool_index_name(IndexNamePrefix.UNIQUE, pending.name, PoolIndexName.CELL),
            *pending_key_columns,
            pending.c[PendingColumn.SAMPLE_IDX],
            unique=True,
        )
        Index(
            pool_index_name(
                IndexNamePrefix.STANDARD, pending.name, PoolIndexName.STATUS_PRIORITY
            ),
            pending.c[PendingColumn.STATUS],
            pending.c[PendingColumn.PRIORITY].desc(),
            pending.c[PendingColumn.CREATED_AT].asc(),
        )
        Index(
            pool_index_name(IndexNamePrefix.STANDARD, pending.name, PoolIndexName.KEY),
            *pending_key_columns,
        )

    def _build_metadata_table(self) -> Table:
        return Table(
            self.schema.table_name(PoolTableType.METADATA),
            self.sa_metadata,
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

    def _build_call_stats_table(self) -> Table:
        return Table(
            self.schema.table_name(PoolTableType.CALL_STATS),
            self.sa_metadata,
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

    def _key_columns(self) -> list[Column[Any]]:
        return [
            Column(key.name, _TYPE_MAP[key.type], nullable=False)
            for key in self.schema.key_columns
        ]
