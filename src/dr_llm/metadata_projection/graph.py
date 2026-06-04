from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.identity import (
    assertion_id,
    content_hash,
    entity_id,
)
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionRole,
    MetadataAssertionType,
    MetadataEntity,
    MetadataEntityType,
    MetadataProjectionError,
    MetadataWritePlan,
)
from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
)


EVENT_ASSERTION_TYPES: dict[StreamingLogEventType, MetadataAssertionType] = {
    StreamingLogEventType.pool_import_started: (
        MetadataAssertionType.pool_import_started
    ),
    StreamingLogEventType.pool_sample_imported: (
        MetadataAssertionType.pool_sample_imported
    ),
    StreamingLogEventType.pool_import_completed: (
        MetadataAssertionType.pool_import_completed
    ),
    StreamingLogEventType.pool_import_failed: (
        MetadataAssertionType.pool_import_failed
    ),
    StreamingLogEventType.work_submitted: MetadataAssertionType.work_submitted,
    StreamingLogEventType.attempt_started: MetadataAssertionType.attempt_started,
    StreamingLogEventType.provider_request_prepared: (
        MetadataAssertionType.provider_request_prepared
    ),
    StreamingLogEventType.provider_response_received: (
        MetadataAssertionType.provider_response_received
    ),
    StreamingLogEventType.attempt_succeeded: (
        MetadataAssertionType.attempt_succeeded
    ),
    StreamingLogEventType.attempt_failed: MetadataAssertionType.attempt_failed,
    StreamingLogEventType.work_retry_scheduled: (
        MetadataAssertionType.work_retry_scheduled
    ),
    StreamingLogEventType.work_completed: MetadataAssertionType.work_completed,
    StreamingLogEventType.work_cancelled: MetadataAssertionType.work_cancelled,
    StreamingLogEventType.producer_started: (
        MetadataAssertionType.producer_started
    ),
    StreamingLogEventType.producer_stopped: (
        MetadataAssertionType.producer_stopped
    ),
    StreamingLogEventType.streaming_log_error: (
        MetadataAssertionType.streaming_log_error
    ),
}


class MetadataGraph:
    def __init__(self) -> None:
        self._entities: dict[str, MetadataEntity] = {}
        self._roles: set[tuple[str, str]] = set()

    def add_entity(
        self,
        entity_type: MetadataEntityType,
        identity_key: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
    ) -> MetadataEntity:
        return self.add_existing_entity(
            metadata_entity(
                entity_type,
                identity_key,
                display_name=display_name,
                metadata=metadata,
                content=content,
            )
        )

    def add_existing_entity(self, entity: MetadataEntity) -> MetadataEntity:
        self._entities.setdefault(entity.entity_id, entity)
        self._roles.add((entity.entity_type, entity.entity_id))
        return entity

    def add_optional_identity(
        self,
        entity_type: MetadataEntityType,
        identity_key: str | None,
    ) -> MetadataEntity | None:
        if identity_key is None:
            return None
        return self.add_entity(entity_type, identity_key)

    def entities(self) -> list[MetadataEntity]:
        return list(self._entities.values())

    def roles_for(self, assertion_id: str) -> list[MetadataAssertionRole]:
        return [
            MetadataAssertionRole(
                assertion_id=assertion_id,
                role_name=role_name,
                entity_id=role_entity_id,
            )
            for role_name, role_entity_id in sorted(self._roles)
        ]


def assertion_type_for_event(
    event_type: StreamingLogEventType,
) -> MetadataAssertionType | None:
    return EVENT_ASSERTION_TYPES.get(event_type)


def source_event_assertion_id(
    config: MetadataProjectionConfig,
    assertion_type: MetadataAssertionType,
    source_idempotency_key: str,
) -> str:
    return assertion_id(
        projection_version=config.projection_version,
        assertion_type=str(assertion_type),
        source_idempotency_key=source_idempotency_key,
    )


def metadata_entity(
    entity_type: MetadataEntityType,
    identity_key: str,
    *,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    content: dict[str, Any] | None = None,
) -> MetadataEntity:
    return MetadataEntity(
        entity_id=entity_id(str(entity_type), identity_key),
        entity_type=str(entity_type),
        identity_key=identity_key,
        content_hash=content_hash(content) if content is not None else None,
        display_name=display_name,
        metadata_json=metadata or {},
    )


def source_event_entity(
    event_id: str,
    event_type: str,
    idempotency_key: str,
) -> MetadataEntity:
    return metadata_entity(
        MetadataEntityType.source_event,
        event_id,
        metadata={
            "event_type": event_type,
            "idempotency_key": idempotency_key,
        },
    )


def producer_entity(producer: ProducerInfo) -> MetadataEntity:
    return _producer_entity(
        name=producer.name,
        version=producer.version,
        instance_id=producer.instance_id,
        metadata=producer.model_dump(mode="json"),
    )


def producer_entity_from_metadata(
    producer: Mapping[str, Any],
) -> MetadataEntity | None:
    name = producer.get("name")
    instance_id = producer.get("instance_id")
    if not isinstance(name, str) or not isinstance(instance_id, str):
        return None
    version = producer.get("version")
    return _producer_entity(
        name=name,
        version=version if isinstance(version, str) else None,
        instance_id=instance_id,
        metadata=dict(producer),
    )


def run_identity_key(event: EventEnvelope) -> str | None:
    payload_run_id = getattr(event.payload, "run_id", None)
    if isinstance(payload_run_id, str):
        return payload_run_id
    return event.run_id


def work_identity_key(event: EventEnvelope) -> str | None:
    payload_work_id = getattr(event.payload, "work_id", None)
    if isinstance(payload_work_id, str):
        return payload_work_id
    return event.work_id


def attempt_identity_key(event: EventEnvelope) -> str | None:
    if event.attempt_id is not None:
        return event.attempt_id
    attempt = getattr(event.payload, "attempt", None)
    resolved_work_id = work_identity_key(event)
    if resolved_work_id is None:
        return None
    if isinstance(attempt, int):
        return f"{resolved_work_id}:{attempt}"
    return f"{resolved_work_id}:attempt"


def merge_write_plans(
    plans: Iterable[MetadataWritePlan],
) -> MetadataWritePlan:
    entities: dict[str, MetadataEntity] = {}
    assertions: dict[str, MetadataAssertion] = {}
    roles: dict[tuple[str, str, str], MetadataAssertionRole] = {}
    errors: list[MetadataProjectionError] = []
    for plan in plans:
        entities.update((entity.entity_id, entity) for entity in plan.entities)
        assertions.update(
            (assertion.assertion_id, assertion)
            for assertion in plan.assertions
        )
        roles.update(
            ((role.assertion_id, role.role_name, role.entity_id), role)
            for role in plan.roles
        )
        errors.extend(plan.errors)
    return MetadataWritePlan(
        entities=list(entities.values()),
        assertions=list(assertions.values()),
        roles=list(roles.values()),
        errors=errors,
    )


def _producer_entity(
    *,
    name: str,
    version: str | None,
    instance_id: str,
    metadata: dict[str, Any],
) -> MetadataEntity:
    identity_key = "|".join([name, version or "", instance_id])
    return metadata_entity(
        MetadataEntityType.producer,
        identity_key,
        display_name=name,
        metadata=metadata,
    )


__all__ = [
    "EVENT_ASSERTION_TYPES",
    "MetadataGraph",
    "assertion_type_for_event",
    "attempt_identity_key",
    "merge_write_plans",
    "metadata_entity",
    "producer_entity",
    "producer_entity_from_metadata",
    "run_identity_key",
    "source_event_assertion_id",
    "source_event_entity",
    "work_identity_key",
]
