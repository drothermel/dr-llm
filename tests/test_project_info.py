from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

import pytest

import dr_llm.project.project_info as project_info_module
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
)
from dr_llm.project.errors import ProjectAlreadyExistsError, ProjectNotFoundError
from dr_llm.project.errors import (
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerPortAllocatedError,
)
from dr_llm.project.project_info import ProjectInfo


def test_create_new_retries_when_docker_reports_port_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_ports: list[int] = []

    monkeypatch.setattr(
        project_info_module,
        "get_all_docker_project_metadata",
        lambda label_prefix: [],
    )

    def fake_create(**kwargs: object) -> None:
        port = kwargs["port"]
        assert isinstance(port, int)
        attempted_ports.append(port)
        if port == 5500:
            raise DockerPortAllocatedError()

    monkeypatch.setattr(project_info_module, "call_docker_create", fake_create)
    monkeypatch.setattr(
        project_info_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )

    project = ProjectInfo.create_new("demo")

    assert attempted_ports == [5500, 5501]
    assert project.port == 5501
    assert project.status == ContainerStatus.RUNNING


def test_create_new_translates_container_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_info_module,
        "get_all_docker_project_metadata",
        lambda label_prefix: [],
    )

    def fake_create(**kwargs: object) -> None:
        _ = kwargs
        raise DockerContainerConflictError()

    monkeypatch.setattr(project_info_module, "call_docker_create", fake_create)

    with pytest.raises(
        ProjectAlreadyExistsError,
        match="Project 'demo' already exists",
    ):
        ProjectInfo.create_new("demo")


def test_get_by_name_raises_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_info_module,
        "get_docker_project_metadata",
        lambda container_name, label_prefix: None,
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        ProjectInfo.get_by_name("demo")


@pytest.mark.parametrize("method_name", ["start", "stop", "destroy"])
def test_direct_operations_translate_missing_container_to_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    def raise_missing_container(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise DockerContainerNotFoundError()

    monkeypatch.setattr(project_info_module, f"call_docker_{method_name}", raise_missing_container)

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        getattr(ProjectInfo(name="demo"), method_name)()


def test_backup_does_not_precheck_cached_running_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_pg_dump_stream(
        *,
        container_name: str,
        db_user: str,
        db_name: str,
        output_stream: IO[bytes],
    ) -> None:
        assert container_name == "dr-llm-pg-demo"
        assert db_user == "postgres"
        assert db_name == "dr_llm"
        output_stream.write(b"select 1;\n")

    monkeypatch.setattr(
        project_info_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    backup_file = ProjectInfo(name="demo", status=ContainerStatus.STOPPED).backup(tmp_path)

    assert backup_file.exists()
    with gzip.open(backup_file, "rb") as f:
        assert f.read() == b"select 1;\n"


def test_backup_translates_missing_container_to_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_pg_dump_stream(
        *,
        container_name: str,
        db_user: str,
        db_name: str,
        output_stream: IO[bytes],
    ) -> None:
        _ = (container_name, db_user, db_name, output_stream)
        raise DockerContainerNotFoundError()

    monkeypatch.setattr(
        project_info_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        ProjectInfo(name="demo").backup(tmp_path)


def test_restore_does_not_precheck_cached_running_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql.gz"
    with gzip.open(backup_file, "wb") as f:
        f.write(b"select 1;\n")
    captured: dict[str, object] = {}

    def fake_swap_in_db(
        *,
        sql_stream: IO[bytes],
        container_name: str,
        db_user: str,
        target_db_name: str,
    ) -> None:
        captured["sql_bytes"] = sql_stream.read()
        captured.update(
            {
                "container_name": container_name,
                "db_user": db_user,
                "target_db_name": target_db_name,
            }
        )

    monkeypatch.setattr(project_info_module, "docker_swap_in_db", fake_swap_in_db)

    ProjectInfo(name="demo", status=ContainerStatus.STOPPED).restore(backup_file)

    assert captured["sql_bytes"] == b"select 1;\n"
    assert captured["container_name"] == "dr-llm-pg-demo"
    assert captured["target_db_name"] == "dr_llm"


def test_restore_translates_missing_container_to_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql.gz"
    with gzip.open(backup_file, "wb") as f:
        f.write(b"select 1;\n")

    def fake_swap_in_db(
        *,
        sql_stream: IO[bytes],
        container_name: str,
        db_user: str,
        target_db_name: str,
    ) -> None:
        _ = (sql_stream, container_name, db_user, target_db_name)
        raise DockerContainerNotFoundError()

    monkeypatch.setattr(project_info_module, "docker_swap_in_db", fake_swap_in_db)

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        ProjectInfo(name="demo").restore(backup_file)


def test_restore_missing_file_raises_native_file_not_found(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "missing.sql.gz"

    with pytest.raises(FileNotFoundError, match=str(backup_file)):
        ProjectInfo(name="demo").restore(backup_file)


def test_restore_rejects_plain_sql_files(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql"
    backup_file.write_text("select 1;\n")

    with pytest.raises(
        ValueError,
        match=r"Restore only supports gzip-compressed SQL backups \(.sql.gz\)\.",
    ):
        ProjectInfo(name="demo").restore(backup_file)


def test_backup_removes_partial_file_when_streaming_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_pg_dump_stream(
        *,
        container_name: str,
        db_user: str,
        db_name: str,
        output_stream: IO[bytes],
    ) -> None:
        assert container_name == "dr-llm-pg-demo"
        assert db_user == "postgres"
        assert db_name == "dr_llm"
        output_stream.write(b"partial dump\n")
        raise RuntimeError("pg_dump failed")

    monkeypatch.setattr(
        project_info_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(RuntimeError, match="pg_dump failed"):
        ProjectInfo(name="demo").backup(tmp_path)

    assert list((tmp_path / "demo").glob("*.sql.gz")) == []
