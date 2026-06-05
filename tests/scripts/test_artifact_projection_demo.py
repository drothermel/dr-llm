from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

from dr_llm.artifact_projection.models import (
    ArtifactIndexSummary,
    ProjectionCheckpoint,
)

runner = CliRunner()


def _load_artifact_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-artifact-projection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_artifact_projection", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_demo_command_forwards_options(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_demo = _load_artifact_demo()
    calls: list[str] = []

    async def fake_run_artifact_demo(options: Any) -> None:
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.artifact_root == tmp_path
        calls.append("run")

    monkeypatch.setattr(
        artifact_demo,
        "_run_artifact_demo",
        fake_run_artifact_demo,
    )

    result = runner.invoke(
        artifact_demo.app,
        [
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--artifact-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]


def test_artifact_summary_accepts_finalized_checkpoint() -> None:
    artifact_demo = _load_artifact_demo()

    artifact_demo._verify_artifact_summary(
        _summary(event_id="event-1"), event_id="event-1"
    )


def test_artifact_summary_rejects_open_references() -> None:
    artifact_demo = _load_artifact_demo()

    with pytest.raises(RuntimeError, match="open references"):
        artifact_demo._verify_artifact_summary(
            _summary(event_id="event-1", open_artifact_count=1),
            event_id="event-1",
        )


def test_artifact_summary_rejects_projection_errors() -> None:
    artifact_demo = _load_artifact_demo()

    with pytest.raises(RuntimeError, match="projection errors"):
        artifact_demo._verify_artifact_summary(
            _summary(event_id="event-1", error_count=1),
            event_id="event-1",
        )


def test_artifact_summary_rejects_checkpoint_mismatch() -> None:
    artifact_demo = _load_artifact_demo()

    with pytest.raises(RuntimeError, match="checkpoint event mismatch"):
        artifact_demo._verify_artifact_summary(
            _summary(event_id="event-2"), event_id="event-1"
        )


def _summary(
    *,
    event_id: str,
    open_artifact_count: int = 0,
    open_shard_count: int = 0,
    error_count: int = 0,
) -> ArtifactIndexSummary:
    return ArtifactIndexSummary(
        artifact_count=1,
        open_artifact_count=open_artifact_count,
        shard_count=1,
        open_shard_count=open_shard_count,
        error_count=error_count,
        checkpoint=ProjectionCheckpoint(
            projection_version="artifact_projection_v1",
            durable_consumer="demo",
            stream_sequence=1,
            event_id=event_id,
        ),
    )
