from __future__ import annotations

from pathlib import Path

import pytest

from dr_llm.artifact_projection import (
    ArtifactLane,
    ArtifactProjectionConfig,
    ArtifactReader,
    ArtifactSourceRef,
    ArtifactStore,
    PayloadArtifactSource,
)
from dr_llm.artifact_projection.index import ArtifactIndexConflictError


def test_store_writes_finalized_artifact_and_rebuilds_index(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    source = _source()

    reference = store.write_artifact(
        source=source, lane=ArtifactLane.json, data=b'{"ok":true}'
    )
    manifest = store.finalize()

    assert manifest is not None
    assert store.index.get_reference(reference.artifact_id) == reference
    assert ArtifactReader(config).read_json(reference) == {"ok": True}

    store.rebuild_index()

    assert store.index.get_reference(reference.artifact_id) == reference


def test_store_sees_open_reference_before_finalize(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()

    reference = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )

    assert (
        store.existing_reference(artifact_id=reference.artifact_id)
        == reference
    )
    assert store.index.get_open_reference(reference.artifact_id) == reference
    assert store.index.get_finalized_reference(reference.artifact_id) is None
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.open_artifact_count == 1
    assert summary.artifact_count == 0


def test_store_duplicate_open_reference_is_noop(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    source = _source()

    first = store.write_artifact(
        source=source, lane=ArtifactLane.text, data=b"hello"
    )
    second = store.write_artifact(
        source=source, lane=ArtifactLane.text, data=b"hello"
    )

    assert second == first
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.open_artifact_count == 1


def test_open_reference_conflict_records_error(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    reference = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )
    conflicting = reference.model_copy(update={"length": 99})

    with pytest.raises(ArtifactIndexConflictError) as excinfo:
        store.insert_open_reference(conflicting)

    assert excinfo.value.artifact_id == reference.artifact_id
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.error_count == 1


def test_store_indexes_rotation_finalized_shard(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(
        artifact_root=tmp_path, target_shard_bytes=5
    )
    store = ArtifactStore(config=config)
    store.initialize()

    first = store.write_artifact(
        source=_source(source_object_key="sha256/ab/one"),
        lane=ArtifactLane.text,
        data=b"hello",
    )
    second = store.write_artifact(
        source=_source(
            source_idempotency_key="idem-2",
            source_object_key="sha256/ab/two",
        ),
        lane=ArtifactLane.text,
        data=b"world",
    )

    assert store.index.get_finalized_reference(first.artifact_id) == first
    assert store.index.get_open_reference(second.artifact_id) == second
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.artifact_count == 1
    assert summary.open_artifact_count == 1
    assert summary.shard_count == 1


def test_rebuild_index_clears_stale_open_references(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    finalized = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )
    store.finalize()
    open_reference = store.write_artifact(
        source=_source(
            source_idempotency_key="idem-2",
            source_object_key="sha256/ab/two",
        ),
        lane=ArtifactLane.text,
        data=b"world",
    )

    store.rebuild_index()

    assert store.index.get_reference(finalized.artifact_id) == finalized
    assert store.index.get_reference(open_reference.artifact_id) is None
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.artifact_count == 1
    assert summary.open_artifact_count == 0


def test_index_duplicate_reference_is_noop(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    reference = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )
    store.finalize()

    assert not store.index.insert_reference(reference)


def test_index_conflicting_reference_raises(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    reference = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )
    store.finalize()
    conflicting = reference.model_copy(update={"length": 99})

    with pytest.raises(ArtifactIndexConflictError) as excinfo:
        store.index.insert_reference(conflicting)

    assert excinfo.value.artifact_id == reference.artifact_id


def _source(
    *,
    source_idempotency_key: str = "idem-1",
    source_object_key: str = "sha256/ab/abc",
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
            size_bytes=12,
            content_type="application/json",
            encoding="utf-8",
            compression="none",
        )
    )
