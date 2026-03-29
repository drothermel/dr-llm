from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import dr_llm.cli.common as cli_common
from dr_llm.cli import app

runner = CliRunner()


class _FakeRepository:
    def start_run(self, **_: object) -> str:
        return "run_cli"

    def upsert_run_parameters(
        self, *, run_id: str, parameters: dict[str, object]
    ) -> int:
        _ = run_id, parameters
        return 1

    def finish_run(self, **_: object) -> None:
        return None

    def list_calls(self, **_: object) -> list[object]:
        return []

    def close(self) -> None:
        return None


def test_run_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _FakeRepository())

    result = runner.invoke(app, ["run", "start"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "parameters_written": 1,
        "run_id": "run_cli",
    }


def test_run_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _FakeRepository())

    result = runner.invoke(
        app, ["run", "finish", "--run-id", "run_cli", "--status", "success"]
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "run_id": "run_cli",
        "status": "success",
    }
