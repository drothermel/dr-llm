from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection._checkpoints import (
    MetadataCheckpointRepository,
)
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import (
    MetadataAssertionType,
    MetadataVerificationResult,
)
from dr_llm.metadata_projection.schema import (
    metadata_assertion_roles,
    metadata_assertions,
    metadata_entities,
)


class MetadataVerificationQueries:
    def __init__(
        self,
        *,
        config: MetadataProjectionConfig,
        checkpoints: MetadataCheckpointRepository,
    ) -> None:
        self.config = config
        self.checkpoints = checkpoints

    def verify(self, conn: Connection) -> MetadataVerificationResult:
        problems: list[str] = []
        problems.extend(self._dangling_role_problems(conn))
        problems.extend(self._missing_checkpoint_problems(conn))
        problems.extend(self._artifact_source_problems(conn))
        return MetadataVerificationResult(
            passed=not problems, problems=problems
        )

    def _dangling_role_problems(self, conn: Connection) -> list[str]:
        dangling_assertions = conn.execute(
            select(func.count())
            .select_from(metadata_assertion_roles)
            .outerjoin(
                metadata_assertions,
                metadata_assertions.c.assertion_id
                == metadata_assertion_roles.c.assertion_id,
            )
            .where(metadata_assertions.c.assertion_id.is_(None))
        ).scalar_one()
        dangling_entities = conn.execute(
            select(func.count())
            .select_from(metadata_assertion_roles)
            .outerjoin(
                metadata_entities,
                metadata_entities.c.entity_id
                == metadata_assertion_roles.c.entity_id,
            )
            .where(metadata_entities.c.entity_id.is_(None))
        ).scalar_one()
        problems: list[str] = []
        if int(dangling_assertions):
            problems.append("roles reference missing assertions")
        if int(dangling_entities):
            problems.append("roles reference missing entities")
        return problems

    def _missing_checkpoint_problems(self, conn: Connection) -> list[str]:
        if (
            self.checkpoints.get(conn, self.config.durable_consumer)
            is not None
        ):
            return []
        return [
            f"missing checkpoint for {self.config.durable_consumer!r}",
        ]

    def _artifact_source_problems(self, conn: Connection) -> list[str]:
        artifact_assertions = metadata_assertions.alias("artifact_assertions")
        source_assertions = metadata_assertions.alias("source_assertions")
        source_event_id = artifact_assertions.c.metadata_json["source_ref"][
            "event_id"
        ].astext
        source_exists = (
            select(1)
            .select_from(source_assertions)
            .where(
                source_assertions.c.projection_version
                == self.config.projection_version,
                source_assertions.c.source_event_id == source_event_id,
                source_assertions.c.assertion_type
                != MetadataAssertionType.artifact_attached,
            )
            .exists()
        )
        missing = conn.execute(
            select(func.count())
            .select_from(artifact_assertions)
            .where(
                artifact_assertions.c.projection_version
                == self.config.projection_version,
                artifact_assertions.c.assertion_type
                == MetadataAssertionType.artifact_attached,
                source_event_id.is_not(None),
                ~source_exists,
            )
        ).scalar_one()
        if missing:
            return [f"{missing} artifact references lack source assertions"]
        return []
