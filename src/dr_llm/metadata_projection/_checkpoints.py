from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import MetadataProjectionCheckpoint
from dr_llm.metadata_projection.schema import metadata_projection_checkpoints


class MetadataCheckpointRepository:
    def __init__(self, config: MetadataProjectionConfig) -> None:
        self.config = config

    def record(
        self,
        conn: Connection,
        checkpoint: MetadataProjectionCheckpoint,
    ) -> None:
        row = checkpoint.model_dump(mode="json")
        stmt = pg_insert(metadata_projection_checkpoints).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["projection_version", "durable_consumer"],
            set_={
                "stream_sequence": stmt.excluded.stream_sequence,
                "event_id": stmt.excluded.event_id,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        conn.execute(stmt)

    def get(
        self, conn: Connection, durable_consumer: str
    ) -> MetadataProjectionCheckpoint | None:
        row = (
            conn.execute(
                select(metadata_projection_checkpoints).where(
                    metadata_projection_checkpoints.c.projection_version
                    == self.config.projection_version,
                    metadata_projection_checkpoints.c.durable_consumer
                    == durable_consumer,
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return MetadataProjectionCheckpoint(**dict(row))
