from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

import pytest

import dr_llm.project.project_info as project_info_module
from dr_llm.project.docker import (
    ContainerStatus,
    DockerContainerConflictError,
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
            raise DockerPortAllocatedError("Docker host port is already allocated.")

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


def test_create_new_propagates_container_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_info_module,
        "get_all_docker_project_metadata",
        lambda label_prefix: [],
    )

    def fake_create(**kwargs: object) -> None:
        _ = kwargs
        raise DockerContainerConflictError("Docker container name is already in use.")

    monkeypatch.setattr(project_info_module, "call_docker_create", fake_create)

    with pytest.raises(
        DockerContainerConflictError,
        match="Docker container name is already in use.",
    ):
        ProjectInfo.create_new("demo")


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
