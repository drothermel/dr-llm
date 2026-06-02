from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from typer.testing import CliRunner

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


def test_artifact_demo_cli_help_smoke() -> None:
    artifact_demo = _load_artifact_demo()

    result = runner.invoke(artifact_demo.app, ["--help"])

    assert result.exit_code == 0


def test_artifact_demo_command_forwards_options(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_demo = _load_artifact_demo()
    calls: list[object] = []

    async def fake_run_artifact_demo(options: object) -> None:
        calls.append(options)

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
    assert len(calls) == 1
    options = calls[0]
    assert isinstance(options, artifact_demo.ArtifactProjectionDemoOptions)
    assert options.nats.nats_url == "nats://localhost:4222"
    assert options.nats.keep_nats
    assert options.artifact_root == tmp_path
