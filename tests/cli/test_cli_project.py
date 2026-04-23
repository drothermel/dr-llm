from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dr_llm.cli.project as project_cli
from dr_llm.cli import app
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.errors import ProjectError
from dr_llm.project.models import (
    CreateProjectRequest,
    DeleteProjectRequest,
    ProjectDeletionResult,
    ProjectDeletionStatus,
)
from dr_llm.project.project_info import ProjectInfo

runner = CliRunner()


def test_project_start_invokes_service_and_reports_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    def fake_start_project(name: str) -> ProjectInfo:
        started.append(name)
        return ProjectInfo(name=name, port=5500, status=ContainerStatus.RUNNING)

    monkeypatch.setattr(project_cli, "start_project", fake_start_project)

    result = runner.invoke(app, ["project", "start", "demo"])

    assert result.exit_code == 0
    assert started == ["demo"]
    assert result.stdout.strip() == "Project 'demo' is running on port 5500"


def test_project_stop_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    def fake_stop_project(name: str) -> None:
        stopped.append(name)

    monkeypatch.setattr(project_cli, "stop_project", fake_stop_project)

    result = runner.invoke(app, ["project", "stop", "demo"])

    assert result.exit_code == 0
    assert stopped == ["demo"]
    assert result.stdout.strip() == "Project 'demo' stopped. Data is preserved."


def test_project_destroy_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    def fake_delete_project(request: DeleteProjectRequest) -> ProjectDeletionResult:
        destroyed.append(request.project_name)
        return ProjectDeletionResult(
            request=request,
            status=ProjectDeletionStatus.deleted,
            destroyed_project_resources=True,
        )

    monkeypatch.setattr(project_cli, "delete_project", fake_delete_project)

    result = runner.invoke(
        app,
        ["project", "destroy", "demo", "--yes-really-delete-everything"],
    )

    assert result.exit_code == 0
    assert destroyed == ["demo"]
    payload = json.loads(result.stdout)
    assert payload["request"]["project_name"] == "demo"
    assert payload["status"] == "deleted"
    assert payload["destroyed_project_resources"] is True


def test_project_backup_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_path = tmp_path / "demo.sql.gz"
    backed_up: list[tuple[str, Path | None]] = []

    def fake_backup_project(name: str, output_dir: Path | None = None) -> Path:
        backed_up.append((name, output_dir))
        return backup_path

    monkeypatch.setattr(project_cli, "backup_project", fake_backup_project)

    result = runner.invoke(
        app,
        ["project", "backup", "demo", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert backed_up == [("demo", tmp_path)]
    assert result.stdout.strip() == f"Backup saved to {backup_path}"


def test_project_restore_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restored: list[tuple[str, Path]] = []
    backup_file = tmp_path / "demo.sql.gz"
    backup_file.write_bytes(b"")

    def fake_restore_project(name: str, backup_path: Path) -> None:
        restored.append((name, backup_path))

    monkeypatch.setattr(project_cli, "restore_project", fake_restore_project)

    result = runner.invoke(
        app,
        ["project", "restore", "demo", str(backup_file)],
    )

    assert result.exit_code == 0
    assert restored == [("demo", backup_file)]
    assert result.stdout.strip() == f"Restored 'demo' from {backup_file}"


def test_project_use_prints_export_for_running_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looked_up: list[str] = []

    def fake_get_project(name: str) -> ProjectInfo:
        looked_up.append(name)
        return ProjectInfo(name=name, port=5500, status=ContainerStatus.RUNNING)

    monkeypatch.setattr(project_cli, "get_project", fake_get_project)

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
    def fake_create_project(request: CreateProjectRequest) -> ProjectInfo:
        assert request.project_name == "demo"
        raise ProjectError("typed project failure")

    monkeypatch.setattr(project_cli, "create_project", fake_create_project)

    result = runner.invoke(app, ["project", "create", "demo"])

    assert result.exit_code == 1
    assert "typed project failure" in result.output


def test_project_list_reports_typed_project_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_list_projects() -> list[ProjectInfo]:
        raise ProjectError("docker unavailable")

    monkeypatch.setattr(project_cli, "list_projects", fake_list_projects)

    result = runner.invoke(app, ["project", "list"])

    assert result.exit_code == 1
    assert "docker unavailable" in result.output


def test_project_backup_reports_file_not_found_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_backup_project(name: str, output_dir: Path | None = None) -> Path:
        _ = (name, output_dir)
        raise FileNotFoundError("missing backup dir")

    monkeypatch.setattr(project_cli, "backup_project", fake_backup_project)

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
