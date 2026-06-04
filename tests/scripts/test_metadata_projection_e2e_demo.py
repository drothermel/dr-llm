from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from typer.testing import CliRunner

runner = CliRunner()


def _load_metadata_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-metadata-projection-e2e.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_metadata_projection_e2e", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_projection_e2e_demo_registers_main_command() -> None:
    metadata_demo = _load_metadata_demo()

    commands = metadata_demo.app.registered_commands

    assert len(commands) == 1
    assert commands[0].callback is metadata_demo.main
    assert commands[0].name is None


def test_metadata_projection_e2e_demo_forwards_options(
    tmp_path: Path, monkeypatch
) -> None:
    metadata_demo = _load_metadata_demo()
    calls: list[str] = []

    async def fake_run_demo(options: Any) -> None:
        assert options.dsn == "postgresql://localhost/demo"
        assert options.project_name == "demo-project"
        assert options.keep_project
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.artifact_root == tmp_path
        assert options.provider == "openai"
        assert options.model == "gpt-test"
        calls.append("run")

    monkeypatch.setattr(metadata_demo, "_run_demo", fake_run_demo)

    result = runner.invoke(
        metadata_demo.app,
        [
            "--dsn",
            "postgresql://localhost/demo",
            "--project-name",
            "demo-project",
            "--keep-project",
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--artifact-root",
            str(tmp_path),
            "--provider",
            "openai",
            "--model",
            "gpt-test",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]
