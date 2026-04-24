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
            self.schema.samples_table,
            self.sa_metadata,
            Column("sample_id", Text, primary_key=True),
            *self._key_columns(),
            Column("sample_idx", Integer, nullable=False),
            Column(
                "payload_json",
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column("source_run_id", Text),
            Column(
                "metadata_json",
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column(
                "created_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )

    def _build_claims_table(self) -> Table:
        return Table(
            self.schema.claims_table,
            self.sa_metadata,
            Column("claim_id", Text, primary_key=True),
            Column("run_id", Text, nullable=False),
            Column("request_id", Text, nullable=False),
            Column("consumer_tag", Text, nullable=False, server_default=text("''")),
            Column("sample_id", Text, nullable=False),
            Column("claim_idx", Integer, nullable=False),
            Column(
                "claimed_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )

    def _build_pending_table(self) -> Table:
        return Table(
            self.schema.pending_table,
            self.sa_metadata,
            Column("pending_id", Text, primary_key=True),
            *self._key_columns(),
            Column("sample_idx", Integer, nullable=False),
            Column(
                "payload_json",
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column("source_run_id", Text),
            Column(
                "metadata_json",
                JSONB,
                nullable=False,
                server_default=text("'{}'::jsonb"),
            ),
            Column("priority", Integer, nullable=False, server_default=text("0")),
            Column("status", Text, nullable=False, server_default=text("'pending'")),
            Column("worker_id", Text),
            Column("lease_expires_at", TIMESTAMP(timezone=True)),
            Column("attempt_count", Integer, nullable=False, server_default=text("0")),
            Column(
                "created_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )

    def _build_indexes(self) -> None:
        Index(
            f"uq_{self.schema.samples_table}_cell",
            *self.samples_key_columns,
            self.samples.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.schema.samples_table}_key",
            *self.samples_key_columns,
        )
        Index(
            f"uq_{self.schema.claims_table}_run_sample",
            self.claims.c.run_id,
            self.claims.c.sample_id,
            unique=True,
        )
        Index(f"idx_{self.schema.claims_table}_run", self.claims.c.run_id)
        Index(
            f"uq_{self.schema.pending_table}_cell",
            *self.pending_key_columns,
            self.pending.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.schema.pending_table}_status_priority",
            self.pending.c.status,
            self.pending.c.priority.desc(),
            self.pending.c.created_at.asc(),
        )
        Index(
            f"idx_{self.schema.pending_table}_key",
            *self.pending_key_columns,
        )

    def _build_metadata_table(self) -> Table:
        return Table(
            self.schema.metadata_table,
            self.sa_metadata,
            Column("pool_name", Text, nullable=False),
            Column("key", Text, nullable=False),
            Column("value_json", JSONB, nullable=False),
            Column(
                "created_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
            Column(
                "updated_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
            PrimaryKeyConstraint("pool_name", "key"),
        )

    def _build_call_stats_table(self) -> Table:
        return Table(
            self.schema.call_stats_table,
            self.sa_metadata,
            Column("sample_id", Text, primary_key=True),
            Column("latency_ms", Integer, nullable=False),
            Column("total_cost_usd", Double),
            Column("prompt_tokens", Integer, nullable=False),
            Column("completion_tokens", Integer, nullable=False),
            Column("reasoning_tokens", Integer),
            Column("total_tokens", Integer, nullable=False),
            Column("attempt_count", Integer, nullable=False, server_default=text("1")),
            Column("finish_reason", Text),
            Column(
                "created_at",
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
