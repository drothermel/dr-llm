from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

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
        "get_claimed_project_ports",
        lambda label_prefix: set(),
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
        "get_claimed_project_ports",
        lambda label_prefix: set(),
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
    def fake_pg_dump(**kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = kwargs
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout=b"select 1;\n",
            stderr=b"",
        )

    monkeypatch.setattr(project_info_module, "call_docker_pg_dump", fake_pg_dump)

    backup_file = ProjectInfo(name="demo", status=ContainerStatus.STOPPED).backup(tmp_path)

    assert backup_file.exists()
    with gzip.open(backup_file, "rb") as f:
        assert f.read() == b"select 1;\n"


def test_restore_does_not_precheck_cached_running_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql"
    backup_file.write_bytes(b"select 1;\n")
    captured: dict[str, object] = {}

    def fake_swap_in_db(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(project_info_module, "docker_swap_in_db", fake_swap_in_db)

    ProjectInfo(name="demo", status=ContainerStatus.STOPPED).restore(backup_file)

    assert captured["sql_bytes"] == b"select 1;\n"
    assert captured["container_name"] == "dr-llm-pg-demo"
    assert captured["target_db_name"] == "dr_llm"


def test_restore_missing_file_raises_native_file_not_found(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "missing.sql"

    with pytest.raises(FileNotFoundError, match=str(backup_file)):
        ProjectInfo(name="demo").restore(backup_file)
