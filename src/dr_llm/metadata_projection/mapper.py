from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.graph import (
    MetadataGraph,
    assertion_type_for_event,
    attempt_identity_key,
    producer_entity,
    run_identity_key,
    source_event_assertion_id,
    source_event_entity,
    work_identity_key,
)
from dr_llm.metadata_projection.identity import content_hash
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionType,
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
        return assertion_type_for_event(event.event_type)

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
        self.graph = MetadataGraph()

    def build(self) -> MetadataWritePlan:
        assertion = self._source_assertion()
        self._add_context_entities()
        self._add_payload_entities()
        return MetadataWritePlan(
            entities=self.graph.entities(),
            assertions=[assertion],
            roles=self.graph.roles_for(assertion.assertion_id),
        )

    def _source_assertion(self) -> MetadataAssertion:
        return MetadataAssertion(
            assertion_id=source_event_assertion_id(
                self.config,
                self.assertion_type,
                self.event.idempotency_key,
            ),
            assertion_type=self.assertion_type,
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

    def _add_context_entities(self) -> None:
        self.graph.add_existing_entity(
            source_event_entity(
                event_id=self.event.event_id,
                event_type=str(self.event.event_type),
                idempotency_key=self.event.idempotency_key,
            )
        )
        self.graph.add_existing_entity(producer_entity(self.event.producer))
        self.graph.add_optional_identity(
            MetadataEntityType.run, run_identity_key(self.event)
        )
        self.graph.add_optional_identity(
            MetadataEntityType.work, work_identity_key(self.event)
        )
        self.graph.add_optional_identity(
            MetadataEntityType.attempt, attempt_identity_key(self.event)
        )

    def _add_payload_entities(self) -> None:
        payload = self.event.payload
        if isinstance(payload, PoolSampleImportedPayload):
            self._add_pool_sample_entities(payload)
        if isinstance(
            payload,
            (
                PoolImportStartedPayload,
                PoolImportCompletedPayload,
                PoolImportFailedPayload,
            ),
        ):
            self._add_pool_entity(payload.pool_name)
        provider = getattr(payload, "provider", None)
        model = getattr(payload, "model", None)
        if isinstance(provider, str):
            self.graph.add_entity(MetadataEntityType.provider, provider)
        if isinstance(provider, str) and isinstance(model, str):
            self._add_model_entity(provider, model)
        self._add_request_summary(getattr(payload, "request_summary", None))
        self._add_response_summary(getattr(payload, "response_summary", None))

    def _add_pool_sample_entities(
        self,
        payload: PoolSampleImportedPayload,
    ) -> None:
        self._add_pool_entity(payload.pool_name)
        identity_key = "|".join(
            [payload.source_id, payload.pool_name, payload.sample_id]
        )
        self.graph.add_entity(
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

    def _add_pool_entity(self, pool_name: str) -> None:
        self.graph.add_entity(
            MetadataEntityType.pool,
            pool_name,
            display_name=pool_name,
        )

    def _add_model_entity(self, provider: str, model: str) -> None:
        self.graph.add_entity(
            MetadataEntityType.model,
            f"{provider}|{model}",
            display_name=model,
            metadata={"provider": provider, "model": model},
        )

    def _add_request_summary(self, summary: RequestSummary | None) -> None:
        if summary is None:
            return
        summary_json = summary.model_dump(mode="json", exclude_none=True)
        self.graph.add_entity(
            MetadataEntityType.model_config,
            content_hash(_model_config_content(summary_json)),
            content=summary_json,
            metadata=_model_config_content(summary_json),
        )
        self.graph.add_entity(
            MetadataEntityType.prompt_instance,
            content_hash(_prompt_content(summary_json)),
            content=_prompt_content(summary_json),
            metadata=_prompt_content(summary_json),
        )

    def _add_response_summary(self, summary: ResponseSummary | None) -> None:
        if summary is None:
            return
        summary_json = summary.model_dump(mode="json", exclude_none=True)
        self.graph.add_entity(
            MetadataEntityType.output_result,
            content_hash(summary_json),
            content=summary_json,
            metadata=summary_json,
        )


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


__all__ = [
    "EventFactMapper",
    "EventWritePlanBuilder",
]
