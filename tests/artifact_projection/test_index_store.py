from __future__ import annotations

from pathlib import Path

import pytest

from dr_llm.artifact_projection import (
    ArtifactLane,
    ArtifactProjectionConfig,
    ArtifactReader,
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


def _source() -> PayloadArtifactSource:
    return PayloadArtifactSource(
        source_event_id="event-1",
        source_event_type="provider_response_received",
        source_schema_version=1,
        source_idempotency_key="idem-1",
        payload_role="response_json",
        source_object_key="sha256/ab/abc",
        source_sha256="a" * 64,
        source_size_bytes=12,
        content_type="application/json",
        encoding="utf-8",
        source_compression="none",
    )
