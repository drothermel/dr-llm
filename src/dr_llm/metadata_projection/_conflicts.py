from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionRole,
    MetadataEntity,
    MetadataProjectionError,
    MetadataProjectionErrorKind,
    MetadataWritePlan,
)
from dr_llm.metadata_projection.schema import (
    metadata_assertion_roles,
    metadata_assertions,
    metadata_entities,
)


class AssertionWriteOutcome(StrEnum):
    inserted = "inserted"
    identical = "identical"
    conflicted = "conflicted"


class MetadataConflictDetector:
    def __init__(self, config: MetadataProjectionConfig) -> None:
        self.config = config

    def entity_conflict_error(
        self,
        conn: Connection,
        entity: MetadataEntity,
        plan: MetadataWritePlan,
    ) -> MetadataProjectionError | None:
        existing = (
            conn.execute(
                select(metadata_entities).where(
                    metadata_entities.c.entity_id == entity.entity_id
                )
            )
            .mappings()
            .first()
        )
        if existing is None:
            return None
        if _entity_stable_fields(existing) == _entity_stable_fields(
            entity.model_dump(mode="python")
        ):
            return None
        return _conflict_error(
            plan,
            MetadataProjectionErrorKind.duplicate_entity_conflict,
            f"Entity {entity.entity_id!r} conflicts",
            self.config.projection_version,
        )

    def existing_assertion_outcome(
        self,
        conn: Connection,
        assertion: MetadataAssertion,
        plan: MetadataWritePlan,
    ) -> AssertionWriteOutcome | None:
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
        if existing is None:
            return None
        return self._assertion_outcome(conn, assertion, existing, plan)

    def assertion_conflict_error(
        self,
        assertion: MetadataAssertion,
        plan: MetadataWritePlan,
    ) -> MetadataProjectionError:
        return _conflict_error(
            plan,
            MetadataProjectionErrorKind.duplicate_assertion_conflict,
            f"Assertion {assertion.assertion_id!r} conflicts",
            self.config.projection_version,
        )

    def _assertion_outcome(
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
