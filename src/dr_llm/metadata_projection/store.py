from __future__ import annotations

from sqlalchemy import delete, select

from dr_llm.metadata_projection._checkpoints import (
    MetadataCheckpointRepository,
)
from dr_llm.metadata_projection._conflicts import MetadataConflictDetector
from dr_llm.metadata_projection._summary import MetadataSummaryQueries
from dr_llm.metadata_projection._verification import (
    MetadataVerificationQueries,
)
from dr_llm.metadata_projection._write_plan import MetadataWritePlanApplier
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import (
    MetadataProjectionCheckpoint,
    MetadataProjectionSummary,
    MetadataVerificationResult,
    MetadataWritePlan,
)
from dr_llm.metadata_projection.schema import (
    create_metadata_projection_schema,
    metadata_assertion_roles,
    metadata_assertions,
    metadata_projection_checkpoints,
    metadata_projection_errors,
)
from dr_llm.pool.db.runtime import DbConfig, DbRuntime


class MetadataStore:
    def __init__(
        self,
        config: MetadataProjectionConfig,
        runtime: DbRuntime | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or DbRuntime(
            DbConfig(
                dsn=config.database_dsn,
                min_pool_size=1,
                max_pool_size=8,
                application_name=config.application_name,
            )
        )
        self._checkpoints = MetadataCheckpointRepository(config)
        self._conflicts = MetadataConflictDetector(config)
        self._write_plans = MetadataWritePlanApplier(
            conflicts=self._conflicts,
            checkpoints=self._checkpoints,
        )
        self._summaries = MetadataSummaryQueries(
            config=config,
            checkpoints=self._checkpoints,
        )
        self._verification = MetadataVerificationQueries(
            config=config,
            checkpoints=self._checkpoints,
        )

    def close(self) -> None:
        self.runtime.close()

    def initialize(self) -> None:
        with self.runtime.begin() as conn:
            create_metadata_projection_schema(conn)

    def apply_write_plan(
        self,
        plan: MetadataWritePlan,
        *,
        checkpoint: MetadataProjectionCheckpoint | None = None,
    ) -> None:
        with self.runtime.begin() as conn:
            self._write_plans.apply(conn, plan, checkpoint=checkpoint)

    def clear_rebuildable_rows(self) -> None:
        with self.runtime.begin() as conn:
            assertion_ids = select(metadata_assertions.c.assertion_id).where(
                metadata_assertions.c.projection_version
                == self.config.projection_version
            )
            conn.execute(
                delete(metadata_assertion_roles).where(
                    metadata_assertion_roles.c.assertion_id.in_(assertion_ids)
                )
            )
            conn.execute(
                delete(metadata_assertions).where(
                    metadata_assertions.c.projection_version
                    == self.config.projection_version
                )
            )
            conn.execute(
                delete(metadata_projection_errors).where(
                    metadata_projection_errors.c.projection_version
                    == self.config.projection_version
                )
            )
            conn.execute(
                delete(metadata_projection_checkpoints).where(
                    metadata_projection_checkpoints.c.projection_version
                    == self.config.projection_version
                )
            )

    def summary(self) -> MetadataProjectionSummary:
        with self.runtime.connect() as conn:
            return self._summaries.summary(conn)

    def verify(self) -> MetadataVerificationResult:
        with self.runtime.connect() as conn:
            return self._verification.verify(conn)


__all__ = ["MetadataStore"]
