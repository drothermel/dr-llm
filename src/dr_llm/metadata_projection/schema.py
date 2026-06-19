from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection


metadata = MetaData()

metadata_entities = Table(
    "metadata_entities",
    metadata,
    Column("entity_id", Text, primary_key=True),
    Column("entity_type", Text, nullable=False),
    Column("identity_key", Text, nullable=False),
    Column("content_hash", Text),
    Column("display_name", Text),
    Column(
        "metadata_json",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    UniqueConstraint(
        "entity_type", "identity_key", name="uq_metadata_entities_identity"
    ),
)

metadata_assertions = Table(
    "metadata_assertions",
    metadata,
    Column("assertion_id", Text, primary_key=True),
    Column("assertion_type", Text, nullable=False),
    Column("projection_version", Text, nullable=False),
    Column("source_event_id", Text, nullable=False),
    Column("source_event_type", Text, nullable=False),
    Column("source_schema_version", Integer, nullable=False),
    Column("source_idempotency_key", Text, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("status", Text),
    Column(
        "metadata_json",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    UniqueConstraint(
        "projection_version",
        "assertion_type",
        "source_idempotency_key",
        name="uq_metadata_assertions_logical",
    ),
)

metadata_assertion_roles = Table(
    "metadata_assertion_roles",
    metadata,
    Column(
        "assertion_id",
        Text,
        ForeignKey("metadata_assertions.assertion_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("role_name", Text, primary_key=True),
    Column(
        "entity_id",
        Text,
        ForeignKey("metadata_entities.entity_id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

metadata_projection_checkpoints = Table(
    "metadata_projection_checkpoints",
    metadata,
    Column("projection_version", Text, primary_key=True),
    Column("durable_consumer", Text, primary_key=True),
    Column("stream_sequence", BigInteger, nullable=False),
    Column("event_id", Text),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
)

metadata_projection_errors = Table(
    "metadata_projection_errors",
    metadata,
    Column("error_id", BigInteger, primary_key=True, autoincrement=True),
    Column("projection_version", Text, nullable=False),
    Column("source_event_id", Text, nullable=False),
    Column("source_idempotency_key", Text, nullable=False),
    Column("source_event_type", Text),
    Column("error_kind", Text, nullable=False),
    Column("message", Text, nullable=False),
    Column(
        "metadata_json",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    ),
    Column("stream_sequence", BigInteger),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
)

Index("ix_metadata_entities_type", metadata_entities.c.entity_type)
Index("ix_metadata_assertions_type", metadata_assertions.c.assertion_type)
Index("ix_metadata_assertions_source", metadata_assertions.c.source_event_id)
Index("ix_metadata_assertions_occurred", metadata_assertions.c.occurred_at)
Index("ix_metadata_errors_kind", metadata_projection_errors.c.error_kind)
Index(
    "ix_metadata_errors_source",
    metadata_projection_errors.c.source_event_id,
)


def create_metadata_projection_schema(conn: Connection) -> None:
    metadata.create_all(bind=conn, checkfirst=True)


__all__ = [
    "create_metadata_projection_schema",
    "metadata",
    "metadata_assertion_roles",
    "metadata_assertions",
    "metadata_entities",
    "metadata_projection_checkpoints",
    "metadata_projection_errors",
]
