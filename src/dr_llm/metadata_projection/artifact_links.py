from __future__ import annotations

from pathlib import Path
from typing import Any

from dr_llm.artifact_projection.index import (
    ArtifactIndex,
    load_manifest_references,
)
from dr_llm.artifact_projection.models import ArtifactReference
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.graph import (
    MetadataGraph,
    merge_write_plans,
    metadata_entity,
    producer_entity_from_metadata,
    source_event_assertion_id,
    source_event_entity,
)
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionType,
    MetadataEntity,
    MetadataEntityType,
    MetadataWritePlan,
)


class ArtifactAttachmentPlanner:
    def __init__(self, config: MetadataProjectionConfig) -> None:
        self.config = config

    def plan_reference(
        self, reference: ArtifactReference
    ) -> MetadataWritePlan:
        assertion = self._assertion(reference)
        graph = self._graph(reference)
        return MetadataWritePlan(
            entities=graph.entities(),
            assertions=[assertion],
            roles=graph.roles_for(assertion.assertion_id),
        )

    def plan_references(
        self, references: list[ArtifactReference]
    ) -> MetadataWritePlan:
        return merge_write_plans(
            self.plan_reference(reference) for reference in references
        )

    def _assertion(self, reference: ArtifactReference) -> MetadataAssertion:
        source_key = artifact_assertion_source_key(reference.artifact_id)
        return MetadataAssertion(
            assertion_id=source_event_assertion_id(
                self.config,
                MetadataAssertionType.artifact_attached,
                source_key,
            ),
            assertion_type=MetadataAssertionType.artifact_attached,
            projection_version=self.config.projection_version,
            source_event_id=reference.source_ref.event_id,
            source_event_type=reference.source_ref.event_type,
            source_schema_version=reference.source_ref.schema_version,
            source_idempotency_key=source_key,
            occurred_at=reference.created_at,
            metadata_json=reference.model_dump(mode="json"),
        )

    def _graph(self, reference: ArtifactReference) -> MetadataGraph:
        graph = MetadataGraph()
        graph.add_existing_entity(self._artifact_entity(reference))
        source_ref = reference.source_ref
        context = reference.event_context
        graph.add_existing_entity(
            source_event_entity(
                source_ref.event_id,
                source_ref.event_type,
                source_ref.idempotency_key,
            )
        )
        graph.add_optional_identity(MetadataEntityType.run, context.run_id)
        graph.add_optional_identity(MetadataEntityType.work, context.work_id)
        graph.add_optional_identity(
            MetadataEntityType.attempt, context.attempt_id
        )
        producer = producer_entity_from_metadata(context.producer)
        if producer is not None:
            graph.add_existing_entity(producer)
        return graph

    def _artifact_entity(self, reference: ArtifactReference) -> MetadataEntity:
        return metadata_entity(
            MetadataEntityType.artifact,
            reference.artifact_id,
            display_name=reference.artifact_id,
            metadata=artifact_entity_metadata(reference),
        )


def artifact_assertion_source_key(artifact_id: str) -> str:
    return f"artifact-v1:artifact_attached:{artifact_id}"


def artifact_entity_metadata(
    reference: ArtifactReference,
) -> dict[str, Any]:
    source_ref = reference.source_ref.model_dump(mode="json")
    event_context = reference.event_context.model_dump(mode="json")
    return {
        "source_ref.event_id": source_ref["event_id"],
        "source_ref.event_type": source_ref["event_type"],
        "source_ref.idempotency_key": source_ref["idempotency_key"],
        "source_ref.payload_role": source_ref["payload_role"],
        "source_ref.object_key": source_ref["object_key"],
        "source_ref.sha256": source_ref["sha256"],
        "source_ref.size_bytes": source_ref["size_bytes"],
        "source_ref.content_type": source_ref["content_type"],
        "source_ref.encoding": source_ref["encoding"],
        "source_ref.compression": source_ref["compression"],
        "event_context.run_id": event_context.get("run_id"),
        "event_context.work_id": event_context.get("work_id"),
        "event_context.attempt_id": event_context.get("attempt_id"),
        "logical_sha256": reference.logical_sha256,
        "size_bytes": reference.size_bytes,
        "lane": str(reference.lane),
        "shard_id": reference.shard_id,
        "shard_uri": reference.shard_uri,
    }


def load_index_references(index_path: Path) -> list[ArtifactReference]:
    with ArtifactIndex(index_path) as index:
        index.initialize()
        return index.list_references()


def load_finalized_manifest_references(
    manifest_paths: list[Path],
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    for path in manifest_paths:
        references.extend(load_manifest_references(path))
    return references


__all__ = [
    "ArtifactAttachmentPlanner",
    "artifact_assertion_source_key",
    "artifact_entity_metadata",
    "load_finalized_manifest_references",
    "load_index_references",
]
