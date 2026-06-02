from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict
import zarr

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.index import (
    dump_json_line,
    load_manifest_references,
    load_shard_manifest,
)
from dr_llm.artifact_projection.models import (
    ArtifactLane,
    ArtifactReference,
    FinalizedShard,
    ShardContents,
    ShardManifest,
)


class ShardStorageBackend(Protocol):
    def initialize(self) -> None: ...

    def publish_shard(
        self, shard: ShardContents, *, writer_session: str
    ) -> None: ...

    def read_bytes(self, reference: ArtifactReference) -> bytes: ...

    def list_finalized_shards(self) -> list[FinalizedShard]: ...


class LocalShardLayout(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    staging_dir: Path
    staging_store: Path
    staging_manifest: Path
    staging_marker: Path
    final_store: Path
    final_manifest: Path
    final_marker: Path

    @classmethod
    def for_shard(
        cls,
        *,
        config: ArtifactProjectionConfig,
        writer_session: str,
        shard_id: str,
    ) -> LocalShardLayout:
        staging_dir = config.staging_root / writer_session / shard_id
        return cls(
            staging_dir=staging_dir,
            staging_store=staging_dir / "store.zarr",
            staging_manifest=staging_dir / "manifest.jsonl",
            staging_marker=staging_dir / "finalized.json",
            final_store=config.shard_root / f"{shard_id}.zarr",
            final_manifest=config.manifest_root / f"{shard_id}.jsonl",
            final_marker=(config.manifest_root / f"{shard_id}.finalized.json"),
        )


class LocalShardStorage:
    def __init__(self, config: ArtifactProjectionConfig) -> None:
        self.config = config

    def initialize(self) -> None:
        self.config.shard_root.mkdir(parents=True, exist_ok=True)
        self.config.manifest_root.mkdir(parents=True, exist_ok=True)
        self.config.staging_root.mkdir(parents=True, exist_ok=True)

    def publish_shard(
        self, shard: ShardContents, *, writer_session: str
    ) -> None:
        layout = self._layout_for(
            shard_id=shard.shard_id, writer_session=writer_session
        )
        _reset_path(layout.staging_dir)
        layout.staging_dir.mkdir(parents=True, exist_ok=True)
        _write_zarr_store(
            store_path=layout.staging_store,
            lanes=shard.lanes,
            chunk_bytes=self.config.chunk_bytes,
        )
        _write_manifest(layout.staging_manifest, shard.references)
        layout.staging_marker.write_text(shard.manifest.model_dump_json())
        _publish_staged_path(layout.staging_store, layout.final_store)
        _publish_staged_path(layout.staging_manifest, layout.final_manifest)
        _publish_staged_path(layout.staging_marker, layout.final_marker)

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        array = self._lane_array(reference)
        start = reference.offset
        stop = start + reference.length
        values = array[start:stop]
        return np.asarray(values, dtype=np.uint8).tobytes()

    def list_finalized_shards(self) -> list[FinalizedShard]:
        finalized_shards: list[FinalizedShard] = []
        for marker_path in self._finalized_marker_paths():
            manifest = load_shard_manifest(marker_path)
            finalized_shards.append(
                FinalizedShard(
                    manifest=manifest,
                    references=load_manifest_references(
                        self._manifest_path(manifest)
                    ),
                )
            )
        return finalized_shards

    def _lane_array(self, reference: ArtifactReference) -> zarr.Array:
        store_path = self.config.layout_root / reference.shard_uri
        group = zarr.open_group(store=str(store_path), mode="r")
        return cast(zarr.Array, group[f"lanes/{reference.lane}"])

    def _layout_for(
        self, *, shard_id: str, writer_session: str
    ) -> LocalShardLayout:
        return LocalShardLayout.for_shard(
            config=self.config,
            writer_session=writer_session,
            shard_id=shard_id,
        )

    def _finalized_marker_paths(self) -> list[Path]:
        return sorted(self.config.manifest_root.glob("*.finalized.json"))

    def _manifest_path(self, manifest: ShardManifest) -> Path:
        return self.config.manifest_root / f"{manifest.shard_id}.jsonl"


class ArtifactReader:
    def __init__(
        self,
        config: ArtifactProjectionConfig,
        *,
        storage: ShardStorageBackend | None = None,
    ) -> None:
        self.storage = storage or LocalShardStorage(config)

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        return self.storage.read_bytes(reference)

    def read_text(self, reference: ArtifactReference) -> str:
        encoding = reference.source_ref.encoding
        if encoding == "binary":
            raise ValueError("binary artifact cannot be decoded as text")
        return self.read_bytes(reference).decode(encoding)

    def read_json(self, reference: ArtifactReference) -> object:
        return json.loads(self.read_text(reference))


def _write_zarr_store(
    *,
    store_path: Path,
    lanes: dict[ArtifactLane, bytes],
    chunk_bytes: int,
) -> None:
    group = zarr.open_group(store=str(store_path), mode="w", zarr_format=3)
    lanes_group = group.create_group("lanes")
    for lane, data in lanes.items():
        values = np.frombuffer(data, dtype=np.uint8)
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
    "LocalShardLayout",
    "LocalShardStorage",
    "ShardStorageBackend",
]
