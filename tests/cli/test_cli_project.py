from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import dr_llm.cli.project as project_cli
from dr_llm.cli import app
from dr_llm.project.errors import ProjectError

runner = CliRunner()


def test_project_start_does_not_lookup_project_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> FakeProjectInfo:
            raise AssertionError(f"unexpected lookup for {name}")

        @classmethod
        def start(cls, name: str) -> None:
            started.append(name)

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(app, ["project", "start", "demo"])

    assert result.exit_code == 0
    assert started == ["demo"]
    assert result.stdout.strip() == "Project 'demo' is running."


def test_project_stop_does_not_lookup_project_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> FakeProjectInfo:
            raise AssertionError(f"unexpected lookup for {name}")

        @classmethod
        def stop(cls, name: str) -> None:
            stopped.append(name)

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(app, ["project", "stop", "demo"])

    assert result.exit_code == 0
    assert stopped == ["demo"]
    assert result.stdout.strip() == "Project 'demo' stopped. Data is preserved."


def test_project_destroy_does_not_lookup_project_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> FakeProjectInfo:
            raise AssertionError(f"unexpected lookup for {name}")

        @classmethod
        def destroy(cls, name: str) -> None:
            destroyed.append(name)

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(
        app,
        ["project", "destroy", "demo", "--yes-really-delete-everything"],
    )

    assert result.exit_code == 0
    assert destroyed == ["demo"]
    assert (
        result.stdout.strip()
        == "Project 'demo' destroyed (container + volume removed)."
    )


def test_project_backup_does_not_lookup_project_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_path = tmp_path / "demo.sql.gz"
    backed_up: list[str] = []

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> FakeProjectInfo:
            raise AssertionError(f"unexpected lookup for {name}")

        @classmethod
        def backup(cls, name: str, output_dir: Path | None = None) -> Path:
            assert output_dir == tmp_path
            backed_up.append(name)
            return backup_path

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(
        app,
        ["project", "backup", "demo", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert backed_up == ["demo"]
    assert result.stdout.strip() == f"Backup saved to {backup_path}"


def test_project_restore_does_not_lookup_project_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restored: list[tuple[str, Path]] = []
    backup_file = tmp_path / "demo.sql.gz"
    backup_file.write_bytes(b"")

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> FakeProjectInfo:
            raise AssertionError(f"unexpected lookup for {name}")

        @classmethod
        def restore(cls, name: str, backup_path: Path) -> None:
            restored.append((name, backup_path))

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(
        app,
        ["project", "restore", "demo", str(backup_file)],
    )

    assert result.exit_code == 0
    assert restored == [("demo", backup_file)]
    assert result.stdout.strip() == f"Restored 'demo' from {backup_file}"


def test_project_use_still_looks_up_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looked_up: list[str] = []

    class FakeProjectInfo:
        @classmethod
        def get_by_name(cls, name: str) -> object:
            looked_up.append(name)
            return type(
                "LookupResult",
                (),
                {
                    "running": True,
                    "dsn": "postgresql://postgres:postgres@localhost:5500/dr_llm",
                },
            )()

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(app, ["project", "use", "demo"])

    assert result.exit_code == 0
    assert looked_up == ["demo"]
    assert (
        result.stdout.strip()
        == "export DR_LLM_DATABASE_URL=postgresql://postgres:postgres@localhost:5500/dr_llm"
    )


def test_project_create_reports_typed_project_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProjectInfo:
        @classmethod
        def create_new(cls, name: str) -> object:
            _ = name
            raise ProjectError("typed project failure")

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(app, ["project", "create", "demo"])

    assert result.exit_code == 1
    assert "typed project failure" in result.output


def test_project_list_reports_typed_project_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProjectInfo:
        @classmethod
        def list_all(cls) -> list[object]:
            raise ProjectError("docker unavailable")

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(app, ["project", "list"])

    assert result.exit_code == 1
    assert "docker unavailable" in result.output


def test_project_backup_reports_file_not_found_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProjectInfo:
        def __init__(self, name: str, **_: object) -> None:
            self.name = name

        def backup(self, output_dir: Path | None = None) -> Path:
            _ = output_dir
            raise FileNotFoundError("missing backup dir")

    monkeypatch.setattr(project_cli, "ProjectInfo", FakeProjectInfo)

    result = runner.invoke(
        app,
        ["project", "backup", "demo", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "missing backup dir" in result.output


def test_global_project_option_is_removed() -> None:
    result = runner.invoke(app, ["--project", "demo", "project", "list"])

    assert result.exit_code != 0
    assert "No such option: --project" in result.output
