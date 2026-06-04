from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionRole,
    MetadataEntity,
    MetadataProjectionCheckpoint,
    MetadataProjectionError,
    MetadataProjectionErrorKind,
    MetadataProjectionSummary,
    MetadataVerificationResult,
    MetadataWritePlan,
)
from dr_llm.metadata_projection.schema import (
    create_metadata_projection_schema,
    metadata_assertion_roles,
    metadata_assertions,
    metadata_entities,
    metadata_projection_checkpoints,
    metadata_projection_errors,
)
from dr_llm.pool.db.runtime import DbConfig, DbRuntime


class AssertionWriteOutcome(StrEnum):
    inserted = "inserted"
    identical = "identical"
    conflicted = "conflicted"


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
            self._apply_write_plan(conn, plan, checkpoint=checkpoint)

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
                checkpoint=self._checkpoint(
                    conn, self.config.durable_consumer
                ),
                artifact_attach_checkpoint=self._checkpoint(
                    conn, self.config.artifact_attach_consumer
                ),
            )

    def verify(self) -> MetadataVerificationResult:
        with self.runtime.connect() as conn:
            problems: list[str] = []
            problems.extend(self._dangling_role_problems(conn))
            problems.extend(self._missing_checkpoint_problems(conn))
            problems.extend(self._artifact_source_problems(conn))
        return MetadataVerificationResult(
            passed=not problems, problems=problems
        )

    def _apply_write_plan(
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
            self._record_checkpoint(conn, checkpoint)

    def _upsert_entity(
        self,
        conn: Connection,
        entity: MetadataEntity,
        plan: MetadataWritePlan,
    ) -> None:
        existing = (
            conn.execute(
                select(metadata_entities).where(
                    metadata_entities.c.entity_id == entity.entity_id
                )
            )
            .mappings()
            .first()
        )
        if existing is not None:
            if _entity_stable_fields(existing) != _entity_stable_fields(
                entity.model_dump(mode="python")
            ):
                self._insert_error(
                    conn,
                    _conflict_error(
                        plan,
                        MetadataProjectionErrorKind.duplicate_entity_conflict,
                        f"Entity {entity.entity_id!r} conflicts",
                        self.config.projection_version,
                    ),
                )
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
        existing = (
            conn.execute(
                select(metadata_assertions).where(
                    metadata_assertions.c.assertion_id
                    == assertion.assertion_id
                )
            )
            .mappings()
            .first()
        )
        if existing is not None:
            return self._record_assertion_conflict_if_needed(
                conn, assertion, existing, plan
            )
        stmt = (
            pg_insert(metadata_assertions)
            .values(assertion.model_dump(mode="json"))
            .on_conflict_do_nothing()
        )
        conn.execute(stmt)
        return AssertionWriteOutcome.inserted

    def _record_assertion_conflict_if_needed(
        self,
        conn: Connection,
        assertion: MetadataAssertion,
        existing: Any,
        plan: MetadataWritePlan,
    ) -> AssertionWriteOutcome:
        stable_fields_match = _assertion_stable_fields(
            existing
        ) == _assertion_stable_fields(assertion.model_dump(mode="python"))
        planned_roles = _roles_for_assertion(
            plan.roles, assertion.assertion_id
        )
        existing_roles = _roles_for_assertion(
            self._existing_roles(conn, assertion.assertion_id),
            assertion.assertion_id,
        )
        if stable_fields_match and existing_roles == planned_roles:
            return AssertionWriteOutcome.identical
        self._insert_error(
            conn,
            _conflict_error(
                plan,
                MetadataProjectionErrorKind.duplicate_assertion_conflict,
                f"Assertion {assertion.assertion_id!r} conflicts",
                self.config.projection_version,
            ),
        )
        return AssertionWriteOutcome.conflicted

    def _existing_roles(
        self, conn: Connection, assertion_id: str
    ) -> list[MetadataAssertionRole]:
        rows = conn.execute(
            select(metadata_assertion_roles).where(
                metadata_assertion_roles.c.assertion_id == assertion_id
            )
        ).mappings()
        return [MetadataAssertionRole(**dict(row)) for row in rows]

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

    def _record_checkpoint(
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

    def _checkpoint(
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
        if self._checkpoint(conn, self.config.durable_consumer) is not None:
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


def _entity_stable_fields(row: Any) -> dict[str, Any]:
    return {
        "entity_type": row["entity_type"],
        "identity_key": row["identity_key"],
        "content_hash": row["content_hash"],
        "display_name": row["display_name"],
        "metadata_json": row["metadata_json"],
    }


def _assertion_stable_fields(row: Any) -> dict[str, Any]:
    return {
        "assertion_type": row["assertion_type"],
        "projection_version": row["projection_version"],
        "source_event_id": row["source_event_id"],
        "source_event_type": row["source_event_type"],
        "source_schema_version": row["source_schema_version"],
        "source_idempotency_key": row["source_idempotency_key"],
        "occurred_at": row["occurred_at"],
        "status": row["status"],
        "metadata_json": row["metadata_json"],
    }


def _roles_for_assertion(
    roles: Iterable[MetadataAssertionRole], assertion_id: str
) -> set[tuple[str, str]]:
    return {
        (role.role_name, role.entity_id)
        for role in roles
        if role.assertion_id == assertion_id
    }


def _conflict_error(
    plan: MetadataWritePlan,
    error_kind: MetadataProjectionErrorKind,
    message: str,
    projection_version: str,
) -> MetadataProjectionError:
    assertion = plan.assertions[0] if plan.assertions else None
    if assertion is None:
        return MetadataProjectionError(
            projection_version=projection_version,
            source_event_id="unknown",
            source_idempotency_key="unknown",
            error_kind=error_kind,
            message=message,
        )
    return MetadataProjectionError(
        projection_version=assertion.projection_version,
        source_event_id=assertion.source_event_id,
        source_idempotency_key=assertion.source_idempotency_key,
        source_event_type=assertion.source_event_type,
        error_kind=error_kind,
        message=message,
    )


def _source_event_id_from_artifact_metadata(
    metadata_json: dict[str, Any],
) -> str | None:
    source_ref = metadata_json.get("source_ref")
    if not isinstance(source_ref, dict):
        return None
    value = source_ref.get("event_id")
    return value if isinstance(value, str) else None


__all__ = ["MetadataStore"]
