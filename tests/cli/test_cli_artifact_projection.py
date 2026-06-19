from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dr_llm.artifact_projection import (
    ArtifactLane,
    ArtifactProjectionConfig,
    ArtifactSourceRef,
    ArtifactStore,
    PayloadArtifactSource,
)
from dr_llm.cli import app

runner = CliRunner()


def test_artifact_read_rejects_open_reference(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "DR_LLM_ARTIFACT_PROJECTION_ARTIFACT_ROOT", str(tmp_path)
    )
    store = ArtifactStore(
        config=ArtifactProjectionConfig(artifact_root=tmp_path)
    )
    store.initialize()
    reference = store.write_artifact(
        source=_source(), lane=ArtifactLane.text, data=b"hello"
    )

    result = runner.invoke(
        app, ["artifact-projection", "read", reference.artifact_id]
    )

    assert result.exit_code != 0
    assert "unknown artifact ID" in result.output


def _source() -> PayloadArtifactSource:
    return PayloadArtifactSource(
        source_ref=ArtifactSourceRef(
            event_id="event-1",
            event_type="provider_response_received",
            schema_version=1,
            idempotency_key="idem-1",
            payload_role="response_json",
            object_key="sha256/ab/abc",
            sha256="a" * 64,
            size_bytes=5,
            content_type="text/plain",
            encoding="utf-8",
            compression="none",
        )
    )
