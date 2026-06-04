from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection._checkpoints import (
    MetadataCheckpointRepository,
)
from dr_llm.metadata_projection._conflicts import (
    AssertionWriteOutcome,
    MetadataConflictDetector,
)
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionRole,
    MetadataEntity,
    MetadataProjectionCheckpoint,
    MetadataProjectionError,
    MetadataWritePlan,
)
from dr_llm.metadata_projection.schema import (
    metadata_assertion_roles,
    metadata_assertions,
    metadata_entities,
    metadata_projection_errors,
)


class MetadataWritePlanApplier:
    def __init__(
        self,
        *,
        conflicts: MetadataConflictDetector,
        checkpoints: MetadataCheckpointRepository,
    ) -> None:
        self.conflicts = conflicts
        self.checkpoints = checkpoints

    def apply(
        self,
        conn: Connection,
        plan: MetadataWritePlan,
        *,
        checkpoint: MetadataProjectionCheckpoint | None,
    ) -> None:
        for entity in plan.entities:
            self._upsert_entity(conn, entity, plan)
        accepted_assertions: set[str] = set()
        for assertion in plan.assertions:
            outcome = self._upsert_assertion(conn, assertion, plan)
            if outcome is not AssertionWriteOutcome.conflicted:
                accepted_assertions.add(assertion.assertion_id)
        for role in plan.roles:
            if role.assertion_id in accepted_assertions:
                self._insert_role(conn, role)
        for error in plan.errors:
            self._insert_error(conn, error)
        if checkpoint is not None:
            self.checkpoints.record(conn, checkpoint)

    def _upsert_entity(
        self,
        conn: Connection,
        entity: MetadataEntity,
        plan: MetadataWritePlan,
    ) -> None:
        conflict_error = self.conflicts.entity_conflict_error(
            conn, entity, plan
        )
        if conflict_error is not None:
            self._insert_error(conn, conflict_error)
            return
        stmt = (
            pg_insert(metadata_entities)
            .values(entity.model_dump(mode="json"))
            .on_conflict_do_nothing()
        )
        conn.execute(stmt)

    def _upsert_assertion(
        self,
        conn: Connection,
        assertion: MetadataAssertion,
        plan: MetadataWritePlan,
    ) -> AssertionWriteOutcome:
        existing_outcome = self.conflicts.existing_assertion_outcome(
            conn, assertion, plan
        )
        if existing_outcome is not None:
            if existing_outcome is AssertionWriteOutcome.conflicted:
                self._insert_error(
                    conn,
                    self.conflicts.assertion_conflict_error(assertion, plan),
                )
            return existing_outcome
        stmt = (
            pg_insert(metadata_assertions)
            .values(assertion.model_dump(mode="json"))
            .on_conflict_do_nothing()
        )
        conn.execute(stmt)
        return AssertionWriteOutcome.inserted

    def _insert_role(
        self, conn: Connection, role: MetadataAssertionRole
    ) -> None:
        stmt = (
            pg_insert(metadata_assertion_roles)
            .values(role.model_dump(mode="json"))
            .on_conflict_do_nothing()
        )
        conn.execute(stmt)

    def _insert_error(
        self, conn: Connection, error: MetadataProjectionError
    ) -> None:
        conn.execute(
            metadata_projection_errors.insert().values(
                error.model_dump(mode="json")
            )
        )
