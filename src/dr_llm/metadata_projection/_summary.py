from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection._checkpoints import (
    MetadataCheckpointRepository,
)
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import MetadataProjectionSummary
from dr_llm.metadata_projection.schema import (
    metadata_assertion_roles,
    metadata_assertions,
    metadata_entities,
    metadata_projection_errors,
)


class MetadataSummaryQueries:
    def __init__(
        self,
        *,
        config: MetadataProjectionConfig,
        checkpoints: MetadataCheckpointRepository,
    ) -> None:
        self.config = config
        self.checkpoints = checkpoints

    def summary(self, conn: Connection) -> MetadataProjectionSummary:
        return MetadataProjectionSummary(
            projection_version=self.config.projection_version,
            entity_count=self._count(conn, metadata_entities),
            assertion_count=self._count(
                conn,
                metadata_assertions,
                projection_version=self.config.projection_version,
            ),
            role_count=self._role_count(conn),
            error_count=self._count(
                conn,
                metadata_projection_errors,
                projection_version=self.config.projection_version,
            ),
            checkpoint=self.checkpoints.get(
                conn, self.config.durable_consumer
            ),
            artifact_attach_checkpoint=self.checkpoints.get(
                conn, self.config.artifact_attach_consumer
            ),
        )

    def _count(
        self,
        conn: Connection,
        table: Any,
        *,
        projection_version: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(table)
        if projection_version is not None:
            stmt = stmt.where(table.c.projection_version == projection_version)
        return int(conn.execute(stmt).scalar_one())

    def _role_count(self, conn: Connection) -> int:
        stmt = (
            select(func.count())
            .select_from(metadata_assertion_roles)
            .join(
                metadata_assertions,
                metadata_assertions.c.assertion_id
                == metadata_assertion_roles.c.assertion_id,
            )
            .where(
                metadata_assertions.c.projection_version
                == self.config.projection_version
            )
        )
        return int(conn.execute(stmt).scalar_one())
