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


def _sqlalchemy_type_for_column_type(column_type: ColumnType) -> type[Any]:
    """Resolve the SQLAlchemy column type class for a schema ``ColumnType``."""
    sql_type = _TYPE_MAP.get(column_type)
    if sql_type is None:
        msg = (
            f"No SQLAlchemy type mapped for ColumnType {column_type!r}; "
            f"add an entry for this value to _TYPE_MAP in dr_llm.pool.db.tables. "
            f"ColumnType={ColumnType!r}, _TYPE_MAP keys={list(_TYPE_MAP)!r}"
        )
        raise ValueError(msg)
    return sql_type


class PoolTables:
    """Dynamic SQLAlchemy table metadata for a pool schema."""

    def __init__(self, schema: PoolSchema) -> None:
        self.schema = schema
        self.metadata = MetaData()
        self.samples = self._build_samples_table()
        self.claims = self._build_claims_table()
        self.pending = self._build_pending_table()
        self.metadata_table = self._build_metadata_table()
        self.samples_key_columns = [
            self.samples.c[name] for name in schema.key_column_names
        ]
        self.pending_key_columns = [
            self.pending.c[name] for name in schema.key_column_names
        ]
        self.all_tables = [
            self.samples,
            self.claims,
            self.pending,
            self.metadata_table,
        ]

    def sample_select_columns(self) -> list[Any]:
        return [
            self.samples.c.sample_id,
            *self.samples_key_columns,
            self.samples.c.sample_idx,
            self.samples.c.payload_json,
            self.samples.c.source_run_id,
            self.samples.c.metadata_json,
            self.samples.c.status,
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
        tbl = Table(
            self.schema.samples_table,
            self.metadata,
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
            Column("status", Text, nullable=False, server_default=text("'active'")),
            Column(
                "created_at",
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("now()"),
            ),
        )
        Index(
            f"uq_{self.schema.samples_table}_cell",
            *[tbl.c[name] for name in self.schema.key_column_names],
            tbl.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.schema.samples_table}_key",
            *[tbl.c[name] for name in self.schema.key_column_names],
        )
        return tbl

    def _build_claims_table(self) -> Table:
        tbl = Table(
            self.schema.claims_table,
            self.metadata,
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
        Index(
            f"uq_{self.schema.claims_table}_run_sample",
            tbl.c.run_id,
            tbl.c.sample_id,
            unique=True,
        )
        Index(f"idx_{self.schema.claims_table}_run", tbl.c.run_id)
        return tbl

    def _build_pending_table(self) -> Table:
        tbl = Table(
            self.schema.pending_table,
            self.metadata,
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
        Index(
            f"uq_{self.schema.pending_table}_cell",
            *[tbl.c[name] for name in self.schema.key_column_names],
            tbl.c.sample_idx,
            unique=True,
        )
        Index(
            f"idx_{self.schema.pending_table}_status_priority",
            tbl.c.status,
            tbl.c.priority.desc(),
            tbl.c.created_at.asc(),
        )
        Index(
            f"idx_{self.schema.pending_table}_key",
            *[tbl.c[name] for name in self.schema.key_column_names],
        )
        return tbl

    def _build_metadata_table(self) -> Table:
        return Table(
            self.schema.metadata_table,
            self.metadata,
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

    def _key_columns(self) -> list[Column[Any]]:
        return [
            Column(
                key.name,
                _sqlalchemy_type_for_column_type(key.type),
                nullable=False,
            )
            for key in self.schema.key_columns
        ]
