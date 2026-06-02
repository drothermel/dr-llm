from __future__ import annotations

from pathlib import Path

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.index import (
    ArtifactIndex,
    ArtifactIndexConflictError,
    load_manifest_references,
    load_shard_manifest,
)
from dr_llm.artifact_projection.models import (
    ArtifactLane,
    ArtifactReference,
    PayloadArtifactSource,
    ProjectionError,
    ProjectionErrorKind,
    ShardManifest,
)
from dr_llm.artifact_projection.shards import ShardWriter


class ArtifactStore:
    def __init__(
        self,
        *,
        config: ArtifactProjectionConfig,
        index: ArtifactIndex | None = None,
        writer: ShardWriter | None = None,
    ) -> None:
        self.config = config
        self.index = index or ArtifactIndex(config.index_path)
        self.writer = writer or ShardWriter(config)

    def initialize(self) -> None:
        self.config.shard_root.mkdir(parents=True, exist_ok=True)
        self.config.manifest_root.mkdir(parents=True, exist_ok=True)
        self.config.staging_root.mkdir(parents=True, exist_ok=True)
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.initialize()

    def existing_reference(
        self, *, artifact_id: str
    ) -> ArtifactReference | None:
        return self.index.get_reference(artifact_id)

    def write_artifact(
        self,
        *,
        source: PayloadArtifactSource,
        lane: ArtifactLane,
        data: bytes,
    ) -> ArtifactReference:
        return self.writer.write_artifact(source=source, lane=lane, data=data)

    def finalize(self) -> ShardManifest | None:
        manifest = self.writer.finalize_current()
        if manifest is None:
            return None
        self.index.insert_shard(manifest)
        for reference in self._manifest_references(manifest):
            self.insert_reference(reference)
        return manifest

    def insert_reference(self, reference: ArtifactReference) -> bool:
        try:
            return self.index.insert_reference(reference)
        except ArtifactIndexConflictError:
            self.index.record_reference_conflict(reference)
            raise

    def rebuild_index(self) -> None:
        self.index.initialize()
        self.index.clear_rebuildable_rows()
        for marker_path in self._finalized_marker_paths():
            manifest = load_shard_manifest(marker_path)
            self.index.insert_shard(manifest)
            for reference in self._manifest_references(manifest):
                self.index.insert_reference(reference)

    def record_error(self, error: ProjectionError) -> None:
        self.index.record_error(error)

    def _manifest_references(
        self, manifest: ShardManifest
    ) -> list[ArtifactReference]:
        manifest_path = (
            self.config.manifest_root / f"{manifest.shard_id}.jsonl"
        )
        return load_manifest_references(manifest_path)

    def _finalized_marker_paths(self) -> list[Path]:
        return sorted(self.config.manifest_root.glob("*.finalized.json"))


def projection_error_for_source(
    *,
    config: ArtifactProjectionConfig,
    source: PayloadArtifactSource,
    error_kind: ProjectionErrorKind,
    message: str,
    stream_sequence: int | None = None,
) -> ProjectionError:
    return ProjectionError(
        projection_version=config.projection_version,
        source_event_id=source.source_event_id,
        source_idempotency_key=source.source_idempotency_key,
        payload_role=source.payload_role,
        source_object_key=source.source_object_key,
        error_kind=error_kind,
        message=message,
        stream_sequence=stream_sequence,
    )


__all__ = ["ArtifactStore", "projection_error_for_source"]
