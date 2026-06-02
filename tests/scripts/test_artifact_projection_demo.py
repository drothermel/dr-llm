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
