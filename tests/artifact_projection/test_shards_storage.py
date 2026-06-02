from __future__ import annotations

from pathlib import Path

import pytest

from dr_llm.artifact_projection import (
    ArtifactEventContext,
    ArtifactLane,
    ArtifactProjectionConfig,
    ArtifactReference,
    ArtifactStore,
    ArtifactSourceRef,
    FinalizedShard,
    PayloadArtifactSource,
    ShardContents,
)
from dr_llm.artifact_projection.shards import OpenShard, ShardWriter
from dr_llm.artifact_projection.storage import (
    ArtifactReader,
    LocalShardStorage,
)


def test_open_shard_builds_contents_without_publishing() -> None:
    shard = OpenShard.create()

    reference = shard.append(
        projection_version="artifact-v1",
        source=_source(),
        lane=ArtifactLane.text,
        data=b"hello",
    )
    contents = shard.build_contents(projection_version="artifact-v1")
    shard.append(
        projection_version="artifact-v1",
        source=_source(
            source_idempotency_key="idem-2",
            source_object_key="sha256/ab/two",
        ),
        lane=ArtifactLane.text,
        data=b"world",
    )

    assert contents.shard_id == shard.shard_id
    assert contents.manifest.shard_id == shard.shard_id
    assert contents.manifest.artifact_count == 1
    assert contents.manifest.lane_sizes == {ArtifactLane.text: 5}
    assert contents.references == [reference]
    assert contents.lanes == {ArtifactLane.text: b"hello"}


def test_open_shard_reference_reuses_source_ref_and_context() -> None:
    shard = OpenShard.create()
    source = _source(
        event_context=ArtifactEventContext(
            run_id="run-1",
            event_source="pool-import",
            metadata={"tenant": "demo"},
        )
    )

    reference = shard.append(
        projection_version="artifact-v1",
        source=source,
        lane=ArtifactLane.text,
        data=b"hello",
    )
    payload = reference.model_dump(mode="json")

    assert reference.source_ref == source.source_ref
    assert reference.event_context == source.event_context
    assert payload["source_ref"]["event_id"] == "event-1"
    assert payload["event_context"]["run_id"] == "run-1"
    assert "source_event_id" not in payload
    assert "source_idempotency_key" not in payload
    assert "event_metadata" not in payload


def test_local_storage_publishes_finalized_shard_and_lists_it(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    storage = LocalShardStorage(config)
    storage.initialize()
    contents = _contents()

    storage.publish_shard(contents, writer_session="writer-1")

    assert (config.shard_root / f"{contents.shard_id}.zarr").is_dir()
    assert (config.manifest_root / f"{contents.shard_id}.jsonl").is_file()
    assert (
        config.manifest_root / f"{contents.shard_id}.finalized.json"
    ).is_file()
    assert ArtifactReader(config).read_text(contents.references[0]) == "hello"
    assert storage.list_finalized_shards() == [
        FinalizedShard(
            manifest=contents.manifest,
            references=contents.references,
        )
    ]


def test_shard_writer_keeps_open_shard_when_publish_fails(
    tmp_path: Path,
) -> None:
    storage = FailingStorage()
    writer = ShardWriter(
        ArtifactProjectionConfig(artifact_root=tmp_path), storage=storage
    )
    writer.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        writer.finalize_current()

    assert writer.current_size == 5


def test_store_rebuilds_index_from_storage_backend(tmp_path: Path) -> None:
    contents = _contents()
    storage = MemoryStorage(
        finalized_shards=[
            FinalizedShard(
                manifest=contents.manifest,
                references=contents.references,
            )
        ]
    )
    store = ArtifactStore(
        config=ArtifactProjectionConfig(artifact_root=tmp_path),
        storage=storage,
    )
    store.initialize()

    store.rebuild_index()

    assert storage.initialized
    assert (
        store.index.get_reference(contents.references[0].artifact_id)
        == contents.references[0]
    )


class FailingStorage:
    def initialize(self) -> None:
        pass

    def publish_shard(
        self, shard: ShardContents, *, writer_session: str
    ) -> None:
        del shard, writer_session
        raise RuntimeError("publish failed")

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        del reference
        raise NotImplementedError

    def list_finalized_shards(self) -> list[FinalizedShard]:
        raise NotImplementedError


class MemoryStorage:
    def __init__(self, *, finalized_shards: list[FinalizedShard]) -> None:
        self.finalized_shards = finalized_shards
        self.initialized = False
        self.published: list[ShardContents] = []

    def initialize(self) -> None:
        self.initialized = True

    def publish_shard(
        self, shard: ShardContents, *, writer_session: str
    ) -> None:
        del writer_session
        self.published.append(shard)

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        del reference
        raise NotImplementedError

    def list_finalized_shards(self) -> list[FinalizedShard]:
        return self.finalized_shards


def _contents() -> ShardContents:
    shard = OpenShard.create()
    shard.append(
        projection_version="artifact-v1",
        source=_source(),
        lane=ArtifactLane.text,
        data=b"hello",
    )
    return shard.build_contents(projection_version="artifact-v1")


def _source(
    *,
    source_idempotency_key: str = "idem-1",
    source_object_key: str = "sha256/ab/abc",
    event_context: ArtifactEventContext | None = None,
) -> PayloadArtifactSource:
    return PayloadArtifactSource(
        source_ref=ArtifactSourceRef(
            event_id="event-1",
            event_type="provider_response_received",
            schema_version=1,
            idempotency_key=source_idempotency_key,
            payload_role="response_json",
            object_key=source_object_key,
            sha256="a" * 64,
            size_bytes=5,
            content_type="text/plain",
            encoding="utf-8",
            compression="none",
        ),
        event_context=event_context or ArtifactEventContext(),
    )
