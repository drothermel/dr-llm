from __future__ import annotations

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
    MetadataColumn,
    PendingColumn,
    PoolTableType,
    SampleColumn,
)
from dr_llm.pool.db.schema import ColumnType, PoolSchema


_TYPE_MAP: dict[ColumnType, type[Any]] = {
    ColumnType.text: Text,
    ColumnType.integer: Integer,
    ColumnType.boolean: Boolean,
    ColumnType.float_: Double,
}


class PoolTables:
    """Dynamic SQLAlchemy table metadata for a pool schema."""

    def __init__(self, schema: PoolSchema) -> None:
        self.schema = schema
        self.sa_metadata = MetaData()
        self.samples = self._build_samples_table()
        self.claims = self._build_claims_table()
        self.pending = self._build_pending_table()
        self.metadata_table = self._build_metadata_table()
        self.call_stats = self._build_call_stats_table()
        self.samples_key_columns = [
            self.samples.c[name] for name in schema.key_column_names
        ]
        self.pending_key_columns = [
            self.pending.c[name] for name in schema.key_column_names
        ]
        self._build_indexes()
        self.all_tables = [
            self.samples,
            self.claims,
            self.pending,
            self.metadata_table,
            self.call_stats,
        ]

    def sample_select_columns(self) -> list[Any]:
        return [
            self.samples.c.sample_id,
            *self.samples_key_columns,
            self.samples.c.sample_idx,
            self.samples.c.payload_json,
            self.samples.c.source_run_id,
            self.samples.c.metadata_json,
            self.samples.c.created_at,
        ]

    def pending_select_columns(self) -> list[Any]:
        return [
            self.pending.c.pending_id,
            *self.pending_key_columns,
            self.pending.c.sample_idx,
            self.pending.c.payload_json,
            self.pending.c.source_run_id,
            self.pending.c.metadata_json,
            self.pending.c.priority,
            self.pending.c.status,
            self.pending.c.worker_id,
            self.pending.c.lease_expires_at,
            self.pending.c.attempt_count,
            self.pending.c.created_at,
        ]

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
                server_default=text("'{}'::jsonb"),
            ),
            Column(SampleColumn.SOURCE_RUN_ID, Text),
            Column(
                SampleColumn.METADATA_JSON,
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column(
                SampleColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
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
                server_default=text("''"),
            ),
            Column(ClaimColumn.SAMPLE_ID, Text, nullable=False),
            Column(ClaimColumn.CLAIM_IDX, Integer, nullable=False),
            Column(
                ClaimColumn.CLAIMED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
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
                server_default=text("'{}'::jsonb"),
            ),
            Column(PendingColumn.SOURCE_RUN_ID, Text),
            Column(
                PendingColumn.METADATA_JSON,
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column(
                PendingColumn.PRIORITY,
                Integer,
                nullable=False,
                server_default=text("0"),
            ),
            Column(
                PendingColumn.STATUS,
                Text,
                nullable=False,
                server_default=text("'pending'"),
            ),
            Column(PendingColumn.WORKER_ID, Text),
            Column(PendingColumn.LEASE_EXPIRES_AT, TIMESTAMP(timezone=True)),
            Column(
                PendingColumn.ATTEMPT_COUNT,
                Integer,
                nullable=False,
                server_default=text("0"),
            ),
            Column(
                PendingColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )

    def _build_indexes(self) -> None:
        Index(
            f"uq_{self.samples.name}_cell",
            *self.samples_key_columns,
            self.samples.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.samples.name}_key",
            *self.samples_key_columns,
        )
        Index(
            f"uq_{self.claims.name}_run_sample",
            self.claims.c.run_id,
            self.claims.c.sample_id,
            unique=True,
        )
        Index(f"idx_{self.claims.name}_run", self.claims.c.run_id)
        Index(
            f"uq_{self.pending.name}_cell",
            *self.pending_key_columns,
            self.pending.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.pending.name}_status_priority",
            self.pending.c.status,
            self.pending.c.priority.desc(),
            self.pending.c.created_at.asc(),
        )
        Index(
            f"idx_{self.pending.name}_key",
            *self.pending_key_columns,
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
                server_default=text("now()"),
            ),
            Column(
                MetadataColumn.UPDATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
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
                server_default=text("1"),
            ),
            Column(CallStatsColumn.FINISH_REASON, Text),
            Column(
                CallStatsColumn.CREATED_AT,
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )

    def _key_columns(self) -> list[Column[Any]]:
        return [
            Column(key.name, _TYPE_MAP[key.type], nullable=False)
            for key in self.schema.key_columns
        ]
