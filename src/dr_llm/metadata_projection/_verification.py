from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection._checkpoints import (
    MetadataCheckpointRepository,
)
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import MetadataVerificationResult
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
        artifact_rows = conn.execute(
            select(metadata_assertions.c.metadata_json).where(
                metadata_assertions.c.projection_version
                == self.config.projection_version,
                metadata_assertions.c.assertion_type == "artifact_attached",
            )
        ).mappings()
        missing = 0
        for row in artifact_rows:
            event_id = _source_event_id_from_artifact_metadata(
                row["metadata_json"]
            )
            if event_id is None:
                continue
            exists = conn.execute(
                select(func.count()).where(
                    metadata_assertions.c.projection_version
                    == self.config.projection_version,
                    metadata_assertions.c.source_event_id == event_id,
                    metadata_assertions.c.assertion_type
                    != "artifact_attached",
                )
            ).scalar_one()
            missing += int(int(exists) == 0)
        if missing:
            return [f"{missing} artifact references lack source assertions"]
        return []


def _source_event_id_from_artifact_metadata(
    metadata_json: dict[str, Any],
) -> str | None:
    source_ref = metadata_json.get("source_ref")
    if not isinstance(source_ref, dict):
        return None
    value = source_ref.get("event_id")
    return value if isinstance(value, str) else None
