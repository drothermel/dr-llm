from __future__ import annotations

from pathlib import Path
from typing import Any

from dr_llm.artifact_projection.index import (
    ArtifactIndex,
    load_manifest_references,
)
from dr_llm.artifact_projection.models import ArtifactReference
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.identity import assertion_id, entity_id
from dr_llm.metadata_projection.mapper import merge_write_plans
from dr_llm.metadata_projection.models import (
    MetadataAssertion,
    MetadataAssertionRole,
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
        entities = self._entities(reference)
        roles = [
            MetadataAssertionRole(
                assertion_id=assertion.assertion_id,
                role_name=entity.entity_type,
                entity_id=entity.entity_id,
            )
            for entity in entities
        ]
        return MetadataWritePlan(
            entities=entities,
            assertions=[assertion],
            roles=roles,
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
            assertion_id=assertion_id(
                projection_version=self.config.projection_version,
                assertion_type=MetadataAssertionType.artifact_attached,
                source_idempotency_key=source_key,
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

    def _entities(self, reference: ArtifactReference) -> list[MetadataEntity]:
        entities = [self._artifact_entity(reference)]
        source_ref = reference.source_ref
        context = reference.event_context
        entities.append(
            _entity(
                MetadataEntityType.source_event,
                source_ref.event_id,
                metadata={
                    "event_type": source_ref.event_type,
                    "idempotency_key": source_ref.idempotency_key,
                },
            )
        )
        entities.extend(
            _optional_identity(MetadataEntityType.run, context.run_id)
        )
        entities.extend(
            _optional_identity(MetadataEntityType.work, context.work_id)
        )
        entities.extend(
            _optional_identity(MetadataEntityType.attempt, context.attempt_id)
        )
        producer = _producer_entity(context.producer)
        if producer is not None:
            entities.append(producer)
        return _unique_entities(entities)

    def _artifact_entity(self, reference: ArtifactReference) -> MetadataEntity:
        return _entity(
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


def _entity(
    entity_type: MetadataEntityType,
    identity_key: str,
    *,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MetadataEntity:
    return MetadataEntity(
        entity_id=entity_id(str(entity_type), identity_key),
        entity_type=str(entity_type),
        identity_key=identity_key,
        display_name=display_name,
        metadata_json=metadata or {},
    )


def _optional_identity(
    entity_type: MetadataEntityType,
    identity_key: str | None,
) -> list[MetadataEntity]:
    if identity_key is None:
        return []
    return [_entity(entity_type, identity_key)]


def _producer_entity(producer: dict[str, Any]) -> MetadataEntity | None:
    name = producer.get("name")
    instance_id = producer.get("instance_id")
    if not isinstance(name, str) or not isinstance(instance_id, str):
        return None
    version = producer.get("version")
    identity_key = "|".join(
        [name, version if isinstance(version, str) else "", instance_id]
    )
    return _entity(
        MetadataEntityType.producer,
        identity_key,
        display_name=name,
        metadata=producer,
    )


def _unique_entities(
    entities: list[MetadataEntity],
) -> list[MetadataEntity]:
    return list({entity.entity_id: entity for entity in entities}.values())


__all__ = [
    "ArtifactAttachmentPlanner",
    "artifact_assertion_source_key",
    "artifact_entity_metadata",
    "load_finalized_manifest_references",
    "load_index_references",
]
