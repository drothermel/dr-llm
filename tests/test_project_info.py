from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

import pytest

import dr_llm.project.project_info as project_info_module
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
)
from dr_llm.project.errors import (
    ProjectAlreadyExistsError,
    ProjectError,
    ProjectNotFoundError,
)
from dr_llm.project.errors import (
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerContainerNotRunningError,
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
        project = kwargs["project"]
        assert isinstance(project, DockerProjectCreateMetadata)
        attempted_ports.append(project.port)
        if project.port == 5500:
            raise DockerPortAllocatedError()

    monkeypatch.setattr(project_info_module, "create_project_container", fake_create)
    monkeypatch.setattr(project_info_module, "_port_has_listener", lambda port: False)
    monkeypatch.setattr(
        project_info_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )

    project = ProjectInfo.create_new("demo")

    assert attempted_ports == [5500, 5501]
    assert project.port == 5501
    assert project.status == ContainerStatus.RUNNING


def test_create_new_skips_ports_with_existing_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_ports: list[int] = []

    monkeypatch.setattr(
        project_info_module,
        "get_all_docker_project_metadata",
        lambda label_prefix: [],
    )
    monkeypatch.setattr(
        project_info_module, "_port_has_listener", lambda port: port == 5500
    )

    def fake_create(**kwargs: object) -> None:
        project = kwargs["project"]
        assert isinstance(project, DockerProjectCreateMetadata)
        attempted_ports.append(project.port)

    monkeypatch.setattr(project_info_module, "create_project_container", fake_create)
    monkeypatch.setattr(
        project_info_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )

    project = ProjectInfo.create_new("demo")

    assert attempted_ports == [5501]
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

    monkeypatch.setattr(project_info_module, "create_project_container", fake_create)
    monkeypatch.setattr(project_info_module, "_port_has_listener", lambda port: False)

    with pytest.raises(
        ProjectAlreadyExistsError,
        match="Project 'demo' already exists",
    ):
        ProjectInfo.create_new("demo")


def test_create_new_cleans_up_container_when_ready_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        project_info_module,
        "get_all_docker_project_metadata",
        lambda label_prefix: [],
    )
    monkeypatch.setattr(project_info_module, "_port_has_listener", lambda port: False)
    monkeypatch.setattr(
        project_info_module, "create_project_container", lambda **kwargs: None
    )

    def fake_wait_docker_ready(**kwargs: object) -> ContainerStatus:
        _ = kwargs
        raise ProjectError("container did not become ready")

    def fake_destroy(container_name: str, volume_name: str) -> None:
        destroyed.append((container_name, volume_name))

    monkeypatch.setattr(
        project_info_module, "wait_docker_ready", fake_wait_docker_ready
    )
    monkeypatch.setattr(project_info_module, "call_docker_destroy", fake_destroy)

    with pytest.raises(ProjectError, match="container did not become ready"):
        ProjectInfo.create_new("demo")

    assert destroyed == [("dr-llm-pg-demo", "dr-llm-data-demo")]


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

    monkeypatch.setattr(
        project_info_module, f"call_docker_{method_name}", raise_missing_container
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        getattr(ProjectInfo, method_name)("demo")


def test_start_returns_fresh_project_metadata_after_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_containers: list[str] = []
    waited_for: list[tuple[str, str, str]] = []

    def fake_start(container_name: str) -> None:
        started_containers.append(container_name)

    def fake_wait_docker_ready(
        *,
        container_name: str,
        db_user: str,
        db_name: str,
    ) -> ContainerStatus:
        waited_for.append((container_name, db_user, db_name))
        return ContainerStatus.RUNNING

    def fake_get_by_name(name: str) -> ProjectInfo:
        return ProjectInfo(name=name, port=5500, status=ContainerStatus.RUNNING)

    monkeypatch.setattr(project_info_module, "call_docker_start", fake_start)
    monkeypatch.setattr(
        project_info_module, "wait_docker_ready", fake_wait_docker_ready
    )
    monkeypatch.setattr(ProjectInfo, "get_by_name", fake_get_by_name)

    project = ProjectInfo.start("demo")

    assert started_containers == ["dr-llm-pg-demo"]
    assert waited_for == [("dr-llm-pg-demo", "postgres", "dr_llm")]
    assert project.port == 5500
    assert project.status == ContainerStatus.RUNNING


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

    backup_file = ProjectInfo.backup("demo", tmp_path)

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
        ProjectInfo.backup("demo", tmp_path)


def test_backup_translates_stopped_container_to_project_error(
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
        raise DockerContainerNotRunningError()

    monkeypatch.setattr(
        project_info_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(
        ProjectError,
        match="Project 'demo' is not running\\. Start it first\\.",
    ):
        ProjectInfo.backup("demo", tmp_path)


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

    ProjectInfo.restore("demo", backup_file)

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
        ProjectInfo.restore("demo", backup_file)


def test_restore_translates_stopped_container_to_project_error(
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
        raise DockerContainerNotRunningError()

    monkeypatch.setattr(project_info_module, "docker_swap_in_db", fake_swap_in_db)

    with pytest.raises(
        ProjectError,
        match="Project 'demo' is not running\\. Start it first\\.",
    ):
        ProjectInfo.restore("demo", backup_file)


def test_restore_missing_file_raises_native_file_not_found(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "missing.sql.gz"

    with pytest.raises(FileNotFoundError, match=str(backup_file)):
        ProjectInfo.restore("demo", backup_file)


def test_restore_rejects_plain_sql_files(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql"
    backup_file.write_text("select 1;\n")

    with pytest.raises(
        ValueError,
        match=r"Restore only supports gzip-compressed SQL backups \(.sql.gz\)\.",
    ):
        ProjectInfo.restore("demo", backup_file)


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
        ProjectInfo.backup("demo", tmp_path)

    assert list((tmp_path / "demo").glob("*.sql.gz")) == []
