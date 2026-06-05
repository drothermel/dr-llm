from __future__ import annotations

from dr_llm.artifact_projection.models import (
    ArtifactEventContext,
    ArtifactLane,
    ArtifactReference,
    ArtifactSourceRef,
)
from dr_llm.metadata_projection import (
    ArtifactAttachmentPlanner,
    EventFactMapper,
    MetadataAssertionType,
    MetadataEntity,
    MetadataEntityType,
    MetadataProjectionConfig,
)
from dr_llm.metadata_projection.graph import EVENT_ASSERTION_TYPES
from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    WorkSubmittedPayload,
)


def test_event_assertion_mapping_covers_streaming_event_types() -> None:
    assert set(EVENT_ASSERTION_TYPES) == set(StreamingLogEventType)
    for event_type, assertion_type in EVENT_ASSERTION_TYPES.items():
        assert assertion_type is MetadataAssertionType(str(event_type))


def test_source_context_entities_match_between_event_and_artifact_plans() -> (
    None
):
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")
    event = EventEnvelope(
        event_id="event-1",
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(
            name="producer", version="1", instance_id="inst"
        ),
        idempotency_key="idem-1",
        payload=WorkSubmittedPayload(
            work_id="work-1",
            run_id="run-1",
            max_retries=0,
        ),
        run_id="run-1",
        work_id="work-1",
    )
    reference = ArtifactReference(
        artifact_id="artifact-1",
        projection_version="artifact-v1",
        source_ref=ArtifactSourceRef(
            event_id=event.event_id,
            event_type=str(event.event_type),
            schema_version=event.schema_version,
            idempotency_key=event.idempotency_key,
            payload_role="response_json",
            object_key="sha256/aa/source",
            sha256="a" * 64,
            size_bytes=10,
            content_type="application/json",
            encoding="utf-8",
            compression="none",
        ),
        event_context=ArtifactEventContext(
            run_id=event.run_id,
            work_id=event.work_id,
            attempt_id=event.attempt_id,
            producer=event.producer.model_dump(mode="json"),
        ),
        logical_sha256="b" * 64,
        size_bytes=10,
        lane=ArtifactLane.json,
        shard_id="shard-1",
        shard_uri="file:///tmp/shard",
        offset=0,
        length=10,
    )

    event_plan = EventFactMapper(config).map_event(event)
    artifact_plan = ArtifactAttachmentPlanner(config).plan_reference(reference)

    for entity_type in (
        MetadataEntityType.source_event,
        MetadataEntityType.producer,
        MetadataEntityType.run,
        MetadataEntityType.work,
    ):
        assert _stable_entity_fields(
            _entity(event_plan.entities, entity_type)
        ) == _stable_entity_fields(
            _entity(artifact_plan.entities, entity_type)
        )


def _entity(
    entities: list[MetadataEntity],
    entity_type: MetadataEntityType,
) -> MetadataEntity:
    return next(
        entity for entity in entities if entity.entity_type == entity_type
    )


def _stable_entity_fields(entity: MetadataEntity) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "identity_key": entity.identity_key,
        "content_hash": entity.content_hash,
        "display_name": entity.display_name,
        "metadata_json": entity.metadata_json,
    }
