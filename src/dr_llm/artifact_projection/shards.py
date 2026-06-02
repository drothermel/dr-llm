from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.identity import (
    artifact_id_for_source,
    sha256_bytes,
)
from dr_llm.artifact_projection.models import (
    ArtifactLane,
    ArtifactReference,
    FinalizedShard,
    PayloadArtifactSource,
    ShardContents,
    ShardManifest,
    ShardWriteResult,
)
from dr_llm.artifact_projection.storage import (
    LocalShardStorage,
    ShardStorageBackend,
)


class ShardWriter:
    def __init__(
        self,
        config: ArtifactProjectionConfig,
        *,
        storage: ShardStorageBackend | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or LocalShardStorage(config)
        self.writer_session = uuid4().hex
        self._current = OpenShard.create()

    @property
    def current_size(self) -> int:
        return self._current.total_size

    def write_artifact(
        self,
        *,
        source: PayloadArtifactSource,
        lane: ArtifactLane,
        data: bytes,
    ) -> ShardWriteResult:
        finalized_shard = None
        if self._should_rotate_for(data):
            finalized_shard = self.finalize_current()
        reference = self._current.append(
            projection_version=self.config.projection_version,
            source=source,
            lane=lane,
            data=data,
        )
        return ShardWriteResult(
            reference=reference, finalized_shard=finalized_shard
        )

    def finalize_current(self) -> FinalizedShard | None:
        if self._current.is_empty:
            return None
        current = self._current
        contents = current.build_contents(
            projection_version=self.config.projection_version
        )
        self.storage.publish_shard(
            contents, writer_session=self.writer_session
        )
        finalized_shard = FinalizedShard(
            manifest=contents.manifest,
            references=contents.references,
        )
        self._current = OpenShard.create()
        return finalized_shard

    def _should_rotate_for(self, data: bytes) -> bool:
        if self._current.is_empty:
            return False
        return (
            self._current.total_size + len(data)
            > self.config.target_shard_bytes
        )


class OpenShard:
    def __init__(self, *, shard_id: str, created_at: datetime) -> None:
        self.shard_id = shard_id
        self.created_at = created_at
        self.references: list[ArtifactReference] = []
        self.lanes: dict[ArtifactLane, bytearray] = {}

    @classmethod
    def create(cls) -> OpenShard:
        return cls(
            shard_id=f"shard_{uuid4().hex}",
            created_at=datetime.now(UTC),
        )

    @property
    def is_empty(self) -> bool:
        return not self.references

    @property
    def total_size(self) -> int:
        return sum(len(buffer) for buffer in self.lanes.values())

    @property
    def shard_uri(self) -> str:
        return f"shards/{self.shard_id}.zarr"

    def append(
        self,
        *,
        projection_version: str,
        source: PayloadArtifactSource,
        lane: ArtifactLane,
        data: bytes,
    ) -> ArtifactReference:
        buffer = self.lanes.setdefault(lane, bytearray())
        offset = len(buffer)
        buffer.extend(data)
        reference = self._reference_for(
            projection_version=projection_version,
            source=source,
            lane=lane,
            data=data,
            offset=offset,
        )
        self.references.append(reference)
        return reference

    def build_contents(self, *, projection_version: str) -> ShardContents:
        return ShardContents(
            shard_id=self.shard_id,
            manifest=self._manifest(projection_version=projection_version),
            references=list(self.references),
            lanes={lane: bytes(buffer) for lane, buffer in self.lanes.items()},
        )

    def _reference_for(
        self,
        *,
        projection_version: str,
        source: PayloadArtifactSource,
        lane: ArtifactLane,
        data: bytes,
        offset: int,
    ) -> ArtifactReference:
        return ArtifactReference(
            artifact_id=artifact_id_for_source(
                projection_version=projection_version,
                source=source,
            ),
            projection_version=projection_version,
            source_event_id=source.source_event_id,
            source_event_type=source.source_event_type,
            source_schema_version=source.source_schema_version,
            source_idempotency_key=source.source_idempotency_key,
            payload_role=source.payload_role,
            source_object_key=source.source_object_key,
            source_sha256=source.source_sha256,
            logical_sha256=sha256_bytes(data),
            size_bytes=len(data),
            content_type=source.content_type,
            encoding=source.encoding,
            source_compression=source.source_compression,
            lane=lane,
            shard_id=self.shard_id,
            shard_uri=self.shard_uri,
            offset=offset,
            length=len(data),
            run_id=source.run_id,
            work_id=source.work_id,
            attempt_id=source.attempt_id,
            causation_id=source.causation_id,
            correlation_id=source.correlation_id,
            source=source.source,
            producer=source.producer,
            event_metadata=source.event_metadata,
        )

    def _manifest(self, *, projection_version: str) -> ShardManifest:
        return ShardManifest(
            shard_id=self.shard_id,
            projection_version=projection_version,
            shard_uri=self.shard_uri,
            artifact_count=len(self.references),
            lane_sizes={
                lane: len(buffer) for lane, buffer in self.lanes.items()
            },
            created_at=self.created_at,
            finalized_at=datetime.now(UTC),
        )


__all__ = [
    "OpenShard",
    "ShardWriter",
]
