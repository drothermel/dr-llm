from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict
import zarr

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.identity import (
    artifact_id_for_source,
    sha256_bytes,
)
from dr_llm.artifact_projection.index import dump_json_line
from dr_llm.artifact_projection.models import (
    ArtifactLane,
    ArtifactReference,
    PayloadArtifactSource,
    ShardManifest,
)


class FinalizedShard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ShardManifest
    references: list[ArtifactReference]


class ShardWriteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: ArtifactReference
    finalized_shard: FinalizedShard | None = None


class ShardWriter:
    def __init__(self, config: ArtifactProjectionConfig) -> None:
        self.config = config
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
        manifest = self._current.finalize(
            config=self.config, writer_session=self.writer_session
        )
        finalized_shard = FinalizedShard(
            manifest=manifest,
            references=list(current.references),
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

    def finalize(
        self,
        *,
        config: ArtifactProjectionConfig,
        writer_session: str,
    ) -> ShardManifest:
        staging_dir = config.staging_root / writer_session / self.shard_id
        final_store = config.shard_root / f"{self.shard_id}.zarr"
        final_manifest = config.manifest_root / f"{self.shard_id}.jsonl"
        final_marker = config.manifest_root / f"{self.shard_id}.finalized.json"
        _reset_path(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        _write_zarr_store(
            store_path=staging_dir / "store.zarr",
            lanes=self.lanes,
            chunk_bytes=config.chunk_bytes,
        )
        manifest = self._manifest(config)
        _write_manifest(staging_dir / "manifest.jsonl", self.references)
        (staging_dir / "finalized.json").write_text(manifest.model_dump_json())
        _publish_staged_path(staging_dir / "store.zarr", final_store)
        _publish_staged_path(staging_dir / "manifest.jsonl", final_manifest)
        _publish_staged_path(staging_dir / "finalized.json", final_marker)
        return manifest

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

    def _manifest(self, config: ArtifactProjectionConfig) -> ShardManifest:
        return ShardManifest(
            shard_id=self.shard_id,
            projection_version=config.projection_version,
            shard_uri=self.shard_uri,
            artifact_count=len(self.references),
            lane_sizes={
                lane: len(buffer) for lane, buffer in self.lanes.items()
            },
            created_at=self.created_at,
            finalized_at=datetime.now(UTC),
        )


class ArtifactReader:
    def __init__(self, config: ArtifactProjectionConfig) -> None:
        self.config = config

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        array = self._lane_array(reference)
        start = reference.offset
        stop = start + reference.length
        values = array[start:stop]
        return np.asarray(values, dtype=np.uint8).tobytes()

    def read_text(self, reference: ArtifactReference) -> str:
        if reference.encoding == "binary":
            raise ValueError("binary artifact cannot be decoded as text")
        return self.read_bytes(reference).decode(reference.encoding)

    def read_json(self, reference: ArtifactReference) -> object:
        return json.loads(self.read_text(reference))

    def _lane_array(self, reference: ArtifactReference) -> zarr.Array:
        store_path = self.config.layout_root / reference.shard_uri
        group = zarr.open_group(store=str(store_path), mode="r")
        return cast(zarr.Array, group[f"lanes/{reference.lane}"])


def _write_zarr_store(
    *,
    store_path: Path,
    lanes: dict[ArtifactLane, bytearray],
    chunk_bytes: int,
) -> None:
    group = zarr.open_group(store=str(store_path), mode="w", zarr_format=3)
    lanes_group = group.create_group("lanes")
    for lane, buffer in lanes.items():
        values = np.frombuffer(bytes(buffer), dtype=np.uint8)
        lane_array = lanes_group.create_array(
            lane,
            shape=(len(values),),
            dtype=np.uint8,
            chunks=(chunk_bytes,),
        )
        lane_array[:] = values


def _write_manifest(path: Path, references: list[ArtifactReference]) -> None:
    with path.open("w") as handle:
        for reference in references:
            handle.write(dump_json_line(reference))


def _publish_staged_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _reset_path(destination)
    os.replace(source, destination)


def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


__all__ = [
    "ArtifactReader",
    "FinalizedShard",
    "OpenShard",
    "ShardWriter",
    "ShardWriteResult",
]
