from __future__ import annotations

from pathlib import Path

from dr_llm.artifact_projection.index import ArtifactIndex
from dr_llm.artifact_projection.models import (
    ArtifactEventContext,
    ArtifactLane,
    ArtifactReference,
    ArtifactSourceRef,
)
from dr_llm.metadata_projection import (
    ArtifactAttachmentPlanner,
    MetadataAssertionType,
    MetadataEntityType,
    MetadataProjectionConfig,
    artifact_assertion_source_key,
    artifact_entity_metadata,
)
from dr_llm.metadata_projection.artifact_links import load_index_references


def test_artifact_attachment_planner_links_finalized_reference() -> None:
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")
    reference = _reference()

    plan = ArtifactAttachmentPlanner(config).plan_reference(reference)

    assert [assertion.assertion_type for assertion in plan.assertions] == [
        MetadataAssertionType.artifact_attached
    ]
    assertion = plan.assertions[0]
    assert assertion.source_idempotency_key == artifact_assertion_source_key(
        "artifact-1"
    )
    entity_types = {entity.entity_type for entity in plan.entities}
    assert MetadataEntityType.artifact in entity_types
    assert MetadataEntityType.source_event in entity_types
    assert MetadataEntityType.run in entity_types
    assert MetadataEntityType.work in entity_types
    assert MetadataEntityType.attempt in entity_types


def test_artifact_entity_metadata_duplicates_query_fields() -> None:
    metadata = artifact_entity_metadata(_reference())

    assert metadata["source_ref.event_id"] == "event-1"
    assert metadata["source_ref.payload_role"] == "response_json"
    assert metadata["event_context.work_id"] == "work-1"
    assert metadata["logical_sha256"] == "b" * 64
    assert metadata["lane"] == "json"


def test_load_index_references_ignores_open_references(tmp_path: Path) -> None:
    index_path = tmp_path / "artifact-index.sqlite"
    with ArtifactIndex(index_path) as index:
        index.initialize()
        index.insert_open_reference(_reference(), writer_session="writer-1")

    assert load_index_references(index_path) == []


def _reference() -> ArtifactReference:
    return ArtifactReference(
        artifact_id="artifact-1",
        projection_version="artifact-v1",
        source_ref=ArtifactSourceRef(
            event_id="event-1",
            event_type="provider_response_received",
            schema_version=1,
            idempotency_key="idem-1",
            payload_role="response_json",
            object_key="sha256/aa/source",
            sha256="a" * 64,
            size_bytes=10,
            content_type="application/json",
            encoding="utf-8",
            compression="none",
        ),
        event_context=ArtifactEventContext(
            run_id="run-1",
            work_id="work-1",
            attempt_id="attempt-1",
            producer={
                "name": "producer",
                "version": "1",
                "instance_id": "inst",
            },
        ),
        logical_sha256="b" * 64,
        size_bytes=10,
        lane=ArtifactLane.json,
        shard_id="shard-1",
        shard_uri="file:///tmp/shard",
        offset=0,
        length=10,
    )
