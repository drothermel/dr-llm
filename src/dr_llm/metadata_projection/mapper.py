from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

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
    MetadataProjectionErrorKind,
    MetadataWritePlan,
)
from dr_llm.streaming_log.events import (
    EventEnvelope,
    PoolImportCompletedPayload,
    PoolImportFailedPayload,
    PoolImportStartedPayload,
    PoolSampleImportedPayload,
    RequestSummary,
    ResponseSummary,
    WorkCancelledPayload,
    WorkSubmittedPayload,
)


class EventFactMapper:
    def __init__(self, config: MetadataProjectionConfig) -> None:
        self.config = config

    def map_event(self, event: EventEnvelope) -> MetadataWritePlan:
        assertion_type = self._assertion_type_for(event)
        if assertion_type is None:
            return MetadataWritePlan(
                errors=[self._unsupported_event_error(event)]
            )
        builder = EventWritePlanBuilder(self.config, event, assertion_type)
        return builder.build()

    def _assertion_type_for(
        self, event: EventEnvelope
    ) -> MetadataAssertionType | None:
        try:
            return MetadataAssertionType(str(event.event_type))
        except ValueError:
            return None

    def _unsupported_event_error(
        self, event: EventEnvelope
    ) -> MetadataProjectionError:
        return MetadataProjectionError(
            projection_version=self.config.projection_version,
            source_event_id=event.event_id,
            source_idempotency_key=event.idempotency_key,
            source_event_type=str(event.event_type),
            error_kind=MetadataProjectionErrorKind.unsupported_event,
            message=f"Unsupported event type {event.event_type!r}",
        )


class EventWritePlanBuilder:
    def __init__(
        self,
        config: MetadataProjectionConfig,
        event: EventEnvelope,
        assertion_type: MetadataAssertionType,
    ) -> None:
        self.config = config
        self.event = event
        self.assertion_type = assertion_type
        self.entities: dict[str, MetadataEntity] = {}
        self.roles: set[tuple[str, str]] = set()

    def build(self) -> MetadataWritePlan:
        assertion = self._source_assertion()
        self._add_context_entities(assertion)
        self._add_payload_entities(assertion)
        return MetadataWritePlan(
            entities=list(self.entities.values()),
            assertions=[assertion],
            roles=[
                MetadataAssertionRole(
                    assertion_id=assertion.assertion_id,
                    role_name=role_name,
                    entity_id=role_entity_id,
                )
                for role_name, role_entity_id in sorted(self.roles)
            ],
        )

    def _source_assertion(self) -> MetadataAssertion:
        return MetadataAssertion(
            assertion_id=assertion_id(
                projection_version=self.config.projection_version,
                assertion_type=str(self.assertion_type),
                source_idempotency_key=self.event.idempotency_key,
            ),
            assertion_type=str(self.assertion_type),
            projection_version=self.config.projection_version,
            source_event_id=self.event.event_id,
            source_event_type=str(self.event.event_type),
            source_schema_version=self.event.schema_version,
            source_idempotency_key=self.event.idempotency_key,
            occurred_at=self.event.occurred_at,
            status=_payload_status(self.event.payload),
            metadata_json=self._assertion_metadata(),
        )

    def _assertion_metadata(self) -> dict[str, Any]:
        return {
            "event": self.event.model_dump(
                mode="json", exclude={"producer", "payload"}
            ),
            "payload": self.event.payload.model_dump(mode="json"),
        }

    def _add_context_entities(self, assertion: MetadataAssertion) -> None:
        self._add_entity(
            assertion,
            MetadataEntityType.source_event,
            self.event.event_id,
            metadata={
                "event_type": str(self.event.event_type),
                "idempotency_key": self.event.idempotency_key,
            },
        )
        self._add_producer(assertion)
        self._add_optional_identity(assertion, MetadataEntityType.run, run_id)
        self._add_optional_identity(
            assertion, MetadataEntityType.work, work_id
        )
        attempt_key = attempt_identity_key(self.event)
        if attempt_key is not None:
            self._add_entity(
                assertion, MetadataEntityType.attempt, attempt_key
            )

    def _add_payload_entities(self, assertion: MetadataAssertion) -> None:
        payload = self.event.payload
        if isinstance(payload, PoolSampleImportedPayload):
            self._add_pool_sample_entities(assertion, payload)
        if isinstance(
            payload,
            (
                PoolImportStartedPayload,
                PoolImportCompletedPayload,
                PoolImportFailedPayload,
            ),
        ):
            self._add_pool_entity(assertion, payload.pool_name)
        provider = getattr(payload, "provider", None)
        model = getattr(payload, "model", None)
        if isinstance(provider, str):
            self._add_entity(assertion, MetadataEntityType.provider, provider)
        if isinstance(provider, str) and isinstance(model, str):
            self._add_model_entity(assertion, provider, model)
        self._add_request_summary(
            assertion, getattr(payload, "request_summary", None)
        )
        self._add_response_summary(
            assertion, getattr(payload, "response_summary", None)
        )

    def _add_producer(self, assertion: MetadataAssertion) -> None:
        producer = self.event.producer
        identity_key = "|".join(
            [producer.name, producer.version or "", producer.instance_id]
        )
        self._add_entity(
            assertion,
            MetadataEntityType.producer,
            identity_key,
            display_name=producer.name,
            metadata=producer.model_dump(mode="json"),
        )

    def _add_optional_identity(
        self,
        assertion: MetadataAssertion,
        entity_type: MetadataEntityType,
        get_identity: Callable[[EventEnvelope], str | None],
    ) -> None:
        identity_key = get_identity(self.event)
        if identity_key is not None:
            self._add_entity(assertion, entity_type, identity_key)

    def _add_pool_sample_entities(
        self,
        assertion: MetadataAssertion,
        payload: PoolSampleImportedPayload,
    ) -> None:
        self._add_pool_entity(assertion, payload.pool_name)
        identity_key = "|".join(
            [payload.source_id, payload.pool_name, payload.sample_id]
        )
        self._add_entity(
            assertion,
            MetadataEntityType.pool_sample,
            identity_key,
            display_name=payload.sample_id,
            metadata={
                "source_id": payload.source_id,
                "pool_name": payload.pool_name,
                "sample_id": payload.sample_id,
                "sample_idx": payload.sample_idx,
                "row_state_hash": payload.row_state_hash,
            },
        )

    def _add_pool_entity(
        self, assertion: MetadataAssertion, pool_name: str
    ) -> None:
        self._add_entity(
            assertion,
            MetadataEntityType.pool,
            pool_name,
            display_name=pool_name,
        )

    def _add_model_entity(
        self, assertion: MetadataAssertion, provider: str, model: str
    ) -> None:
        self._add_entity(
            assertion,
            MetadataEntityType.model,
            f"{provider}|{model}",
            display_name=model,
            metadata={"provider": provider, "model": model},
        )

    def _add_request_summary(
        self, assertion: MetadataAssertion, summary: RequestSummary | None
    ) -> None:
        if summary is None:
            return
        summary_json = summary.model_dump(mode="json", exclude_none=True)
        self._add_entity(
            assertion,
            MetadataEntityType.model_config,
            content_hash(_model_config_content(summary_json)),
            content=summary_json,
            metadata=_model_config_content(summary_json),
        )
        self._add_entity(
            assertion,
            MetadataEntityType.prompt_instance,
            content_hash(_prompt_content(summary_json)),
            content=summary_json,
            metadata=_prompt_content(summary_json),
        )

    def _add_response_summary(
        self, assertion: MetadataAssertion, summary: ResponseSummary | None
    ) -> None:
        if summary is None:
            return
        summary_json = summary.model_dump(mode="json", exclude_none=True)
        self._add_entity(
            assertion,
            MetadataEntityType.output_result,
            content_hash(summary_json),
            content=summary_json,
            metadata=summary_json,
        )

    def _add_entity(
        self,
        assertion: MetadataAssertion,
        entity_type: MetadataEntityType,
        identity_key: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
    ) -> MetadataEntity:
        resolved_entity_id = entity_id(str(entity_type), identity_key)
        entity = MetadataEntity(
            entity_id=resolved_entity_id,
            entity_type=str(entity_type),
            identity_key=identity_key,
            content_hash=content_hash(content)
            if content is not None
            else None,
            display_name=display_name,
            metadata_json=metadata or {},
        )
        self.entities.setdefault(resolved_entity_id, entity)
        self.roles.add((str(entity_type), resolved_entity_id))
        return entity


def run_id(event: EventEnvelope) -> str | None:
    payload_run_id = getattr(event.payload, "run_id", None)
    if isinstance(payload_run_id, str):
        return payload_run_id
    return event.run_id


def work_id(event: EventEnvelope) -> str | None:
    payload_work_id = getattr(event.payload, "work_id", None)
    if isinstance(payload_work_id, str):
        return payload_work_id
    return event.work_id


def attempt_identity_key(event: EventEnvelope) -> str | None:
    if event.attempt_id is not None:
        return event.attempt_id
    attempt = getattr(event.payload, "attempt", None)
    resolved_work_id = work_id(event)
    if resolved_work_id is None:
        return None
    if isinstance(attempt, int):
        return f"{resolved_work_id}:{attempt}"
    return f"{resolved_work_id}:attempt"


def _payload_status(payload: BaseModel) -> str | None:
    value = getattr(payload, "status", None)
    if isinstance(value, str):
        return value
    if isinstance(payload, WorkCancelledPayload):
        return "cancelled"
    if isinstance(payload, WorkSubmittedPayload):
        return "submitted"
    return None


def _model_config_content(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "provider",
            "model",
            "mode",
            "max_tokens",
            "effort",
            "sampling",
        )
        if key in summary
    }


def _prompt_content(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in ("messages_sha256", "message_count", "prompt_preview")
        if key in summary
    }


def merge_write_plans(plans: Iterable[MetadataWritePlan]) -> MetadataWritePlan:
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


__all__ = [
    "EventFactMapper",
    "EventWritePlanBuilder",
    "attempt_identity_key",
    "merge_write_plans",
    "run_id",
    "work_id",
]
