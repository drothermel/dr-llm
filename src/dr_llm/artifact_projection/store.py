from __future__ import annotations

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.index import (
    ArtifactIndex,
    ArtifactIndexConflictError,
)
from dr_llm.artifact_projection.identity import artifact_id_for_source_ref
from dr_llm.artifact_projection.models import (
    ArtifactLane,
    ArtifactReference,
    FinalizedShard,
    PayloadArtifactSource,
    ProjectionError,
    ProjectionErrorKind,
    ShardManifest,
)
from dr_llm.artifact_projection.shards import ShardWriter
from dr_llm.artifact_projection.storage import (
    LocalShardStorage,
    ShardStorageBackend,
)


class ArtifactStore:
    def __init__(
        self,
        *,
        config: ArtifactProjectionConfig,
        index: ArtifactIndex | None = None,
        writer: ShardWriter | None = None,
        storage: ShardStorageBackend | None = None,
    ) -> None:
        self.config = config
        self.index = index or ArtifactIndex(config.index_path)
        self.storage = storage or (
            writer.storage if writer is not None else LocalShardStorage(config)
        )
        if writer is not None and storage is not None:
            if writer.storage is not storage:
                raise ValueError(
                    "writer and storage must use the same backend"
                )
        self.writer = writer or ShardWriter(config, storage=self.storage)

    def initialize(self) -> None:
        self.storage.initialize()
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
        existing = self.existing_reference(
            artifact_id=self._artifact_id_for_source(source)
        )
        if existing is not None:
            return existing
        result = self.writer.write_artifact(
            source=source, lane=lane, data=data
        )
        if result.finalized_shard is not None:
            self._commit_finalized_shard(result.finalized_shard)
        self.insert_open_reference(result.reference)
        return result.reference

    def finalize(self) -> ShardManifest | None:
        finalized_shard = self.writer.finalize_current()
        if finalized_shard is None:
            return None
        self._commit_finalized_shard(finalized_shard)
        return finalized_shard.manifest

    def insert_reference(self, reference: ArtifactReference) -> bool:
        try:
            return self.index.insert_reference(reference)
        except ArtifactIndexConflictError:
            self.index.record_reference_conflict(reference)
            raise

    def insert_open_reference(self, reference: ArtifactReference) -> bool:
        try:
            return self.index.insert_open_reference(
                reference, writer_session=self.writer.writer_session
            )
        except ArtifactIndexConflictError:
            self.index.record_reference_conflict(reference)
            raise

    def rebuild_index(self) -> None:
        self.index.initialize()
        self.index.clear_rebuildable_rows()
        for finalized_shard in self.storage.list_finalized_shards():
            manifest = finalized_shard.manifest
            self.index.insert_shard(manifest)
            for reference in finalized_shard.references:
                self.index.insert_reference(reference)

    def record_error(self, error: ProjectionError) -> None:
        self.index.record_error(error)

    def _commit_finalized_shard(self, finalized_shard: FinalizedShard) -> None:
        try:
            self.index.finalize_shard_references(
                finalized_shard.manifest, finalized_shard.references
            )
        except ArtifactIndexConflictError as exc:
            reference = next(
                reference
                for reference in finalized_shard.references
                if reference.artifact_id == exc.artifact_id
            )
            self.index.record_reference_conflict(reference)
            raise

    def _artifact_id_for_source(self, source: PayloadArtifactSource) -> str:
        return artifact_id_for_source_ref(
            projection_version=self.config.projection_version,
            source_ref=source.source_ref,
        )


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
        source_ref=source.source_ref,
        event_context=source.event_context,
        error_kind=error_kind,
        message=message,
        stream_sequence=stream_sequence,
    )


__all__ = ["ArtifactStore", "projection_error_for_source"]
