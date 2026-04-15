from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

import pytest

import dr_llm.project.project_service as project_service_module
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
)
from dr_llm.project.errors import (
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerContainerNotRunningError,
    DockerPortAllocatedError,
    ProjectAlreadyExistsError,
    ProjectError,
    ProjectNotFoundError,
)
from dr_llm.project.models import (
    CreateProjectRequest,
    ProjectCreationBlockReason,
)
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project.project_service import (
    assess_project_creation,
    backup_project,
    create_project,
    get_project,
    inspect_projects,
    restore_project,
    start_project,
)


def test_create_project_retries_when_docker_reports_port_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_ports: list[int] = []

    monkeypatch.setattr(
        project_service_module,
        "get_all_docker_project_metadata",
        list,
    )

    def fake_create(**kwargs: object) -> None:
        project = kwargs["project"]
        assert isinstance(project, DockerProjectCreateMetadata)
        attempted_ports.append(project.port)
        if project.port == 5500:
            raise DockerPortAllocatedError()

    monkeypatch.setattr(project_service_module, "create_project_container", fake_create)
    monkeypatch.setattr(
        project_service_module, "_port_has_listener", lambda port: False
    )
    monkeypatch.setattr(
        project_service_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )
    monkeypatch.setattr(project_service_module, "wait_dsn_ready", lambda dsn: None)

    project = create_project(CreateProjectRequest(project_name="demo"))

    assert attempted_ports == [5500, 5501]
    assert project.port == 5501
    assert project.status == ContainerStatus.RUNNING


def test_create_project_skips_ports_with_existing_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_ports: list[int] = []

    monkeypatch.setattr(
        project_service_module,
        "get_all_docker_project_metadata",
        list,
    )
    monkeypatch.setattr(
        project_service_module, "_port_has_listener", lambda port: port == 5500
    )

    def fake_create(**kwargs: object) -> None:
        project = kwargs["project"]
        assert isinstance(project, DockerProjectCreateMetadata)
        attempted_ports.append(project.port)

    monkeypatch.setattr(project_service_module, "create_project_container", fake_create)
    monkeypatch.setattr(
        project_service_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )
    monkeypatch.setattr(project_service_module, "wait_dsn_ready", lambda dsn: None)

    project = create_project(CreateProjectRequest(project_name="demo"))

    assert attempted_ports == [5501]
    assert project.port == 5501
    assert project.status == ContainerStatus.RUNNING


def test_create_project_translates_container_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_service_module,
        "get_all_docker_project_metadata",
        list,
    )

    def fake_create(**kwargs: object) -> None:
        _ = kwargs
        raise DockerContainerConflictError()

    monkeypatch.setattr(project_service_module, "create_project_container", fake_create)
    monkeypatch.setattr(
        project_service_module, "_port_has_listener", lambda port: False
    )

    with pytest.raises(
        ProjectAlreadyExistsError,
        match="Project 'demo' already exists",
    ):
        create_project(CreateProjectRequest(project_name="demo"))


def test_create_project_cleans_up_container_when_ready_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        project_service_module,
        "get_all_docker_project_metadata",
        list,
    )
    monkeypatch.setattr(
        project_service_module, "_port_has_listener", lambda port: False
    )
    monkeypatch.setattr(
        project_service_module, "create_project_container", lambda **kwargs: None
    )

    def fake_wait_docker_ready(**kwargs: object) -> ContainerStatus:
        _ = kwargs
        raise ProjectError("container did not become ready")

    def fake_destroy(container_name: str, volume_name: str) -> None:
        destroyed.append((container_name, volume_name))

    monkeypatch.setattr(
        project_service_module, "wait_docker_ready", fake_wait_docker_ready
    )
    monkeypatch.setattr(project_service_module, "call_docker_destroy", fake_destroy)

    with pytest.raises(ProjectError, match="container did not become ready"):
        create_project(CreateProjectRequest(project_name="demo"))

    assert destroyed == [("dr-llm-pg-demo", "dr-llm-data-demo")]


def test_create_project_cleans_up_container_when_dsn_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the host-side DSN probe fails after wait_docker_ready succeeds,
    the freshly-created container must still be destroyed so callers don't
    leak Docker resources on a failed startup."""
    destroyed: list[tuple[str, str]] = []
    probed_dsns: list[str] = []

    monkeypatch.setattr(
        project_service_module,
        "get_all_docker_project_metadata",
        list,
    )
    monkeypatch.setattr(
        project_service_module, "_port_has_listener", lambda port: False
    )
    monkeypatch.setattr(
        project_service_module, "create_project_container", lambda **kwargs: None
    )
    monkeypatch.setattr(
        project_service_module,
        "wait_docker_ready",
        lambda **kwargs: ContainerStatus.RUNNING,
    )

    def fake_wait_dsn_ready(dsn: str) -> None:
        probed_dsns.append(dsn)
        raise ProjectError("postgres did not accept SQL connections")

    monkeypatch.setattr(project_service_module, "wait_dsn_ready", fake_wait_dsn_ready)
    monkeypatch.setattr(
        project_service_module,
        "call_docker_destroy",
        lambda container_name, volume_name: destroyed.append(
            (container_name, volume_name)
        ),
    )

    with pytest.raises(ProjectError, match="did not accept SQL connections"):
        create_project(CreateProjectRequest(project_name="demo"))

    assert probed_dsns == ["postgresql://postgres:postgres@localhost:5500/dr_llm"]
    assert destroyed == [("dr-llm-pg-demo", "dr-llm-data-demo")]


def test_assess_project_creation_reports_invalid_name_existing_project_and_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recent = ProjectInfo(
        name="demo",
        port=5500,
        created_at=project_service_module.datetime.now(project_service_module.UTC),
    )
    monkeypatch.setattr(project_service_module, "list_projects", lambda: [recent])

    readiness = assess_project_creation(CreateProjectRequest(project_name="Demo"))

    assert readiness.allowed is False
    assert {violation.reason for violation in readiness.violations} == {
        ProjectCreationBlockReason.invalid_name,
        ProjectCreationBlockReason.cooldown_active,
    }


def test_assess_project_creation_reports_duplicate_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_service_module,
        "list_projects",
        lambda: [ProjectInfo(name="demo", port=5500)],
    )

    readiness = assess_project_creation(CreateProjectRequest(project_name="demo"))

    assert readiness.allowed is False
    assert [violation.reason for violation in readiness.violations] == [
        ProjectCreationBlockReason.already_exists
    ]


def test_inspect_projects_includes_discovered_pool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = [
        ProjectInfo(name="running", port=5500, status=ContainerStatus.RUNNING),
        ProjectInfo(name="stopped", status=ContainerStatus.STOPPED),
    ]
    monkeypatch.setattr(project_service_module, "list_projects", lambda: projects)
    monkeypatch.setattr(
        "dr_llm.pool.admin_service.discover_pools",
        lambda dsn: ["alpha", "beta"] if dsn.endswith(":5500/dr_llm") else [],
    )

    summaries = inspect_projects()

    assert [
        (summary.project.name, summary.pool_names, summary.pool_count)
        for summary in summaries
    ] == [
        ("running", ["alpha", "beta"], 2),
        ("stopped", [], 0),
    ]


def test_inspect_projects_tolerates_pool_discovery_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="running", port=5500, status=ContainerStatus.RUNNING)
    monkeypatch.setattr(project_service_module, "list_projects", lambda: [project])
    monkeypatch.setattr(
        "dr_llm.pool.admin_service.discover_pools",
        lambda dsn: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    summaries = inspect_projects()

    assert [
        (summary.project.name, summary.pool_names, summary.pool_count)
        for summary in summaries
    ] == [("running", [], 0)]


def test_get_project_raises_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_service_module,
        "get_docker_project_metadata",
        lambda container_name: None,
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        get_project("demo")


@pytest.mark.parametrize(
    ("function_name", "docker_call_name"),
    [
        ("start_project", "call_docker_start"),
        ("stop_project", "call_docker_stop"),
        ("destroy_project", "call_docker_destroy"),
    ],
)
def test_direct_operations_translate_missing_container_to_project_not_found(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    docker_call_name: str,
) -> None:
    def raise_missing_container(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise DockerContainerNotFoundError()

    monkeypatch.setattr(
        project_service_module, docker_call_name, raise_missing_container
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        getattr(project_service_module, function_name)("demo")


def test_start_project_returns_fresh_project_metadata_after_ready(
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

    def fake_get_project(name: str) -> ProjectInfo:
        return ProjectInfo(name=name, port=5500, status=ContainerStatus.RUNNING)

    probed_dsns: list[str] = []

    def fake_wait_dsn_ready(dsn: str) -> None:
        probed_dsns.append(dsn)

    monkeypatch.setattr(project_service_module, "call_docker_start", fake_start)
    monkeypatch.setattr(
        project_service_module, "wait_docker_ready", fake_wait_docker_ready
    )
    monkeypatch.setattr(project_service_module, "wait_dsn_ready", fake_wait_dsn_ready)
    monkeypatch.setattr(project_service_module, "get_project", fake_get_project)

    project = start_project("demo")

    assert started_containers == ["dr-llm-pg-demo"]
    assert waited_for == [("dr-llm-pg-demo", "postgres", "dr_llm")]
    assert probed_dsns == ["postgresql://postgres:postgres@localhost:5500/dr_llm"]
    assert project.port == 5500
    assert project.status == ContainerStatus.RUNNING


def test_backup_project_does_not_precheck_cached_running_status(
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
        project_service_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    backup_file = backup_project("demo", tmp_path)

    assert backup_file.exists()
    with gzip.open(backup_file, "rb") as f:
        assert f.read() == b"select 1;\n"


def test_backup_project_translates_missing_container_to_project_not_found(
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
        project_service_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        backup_project("demo", tmp_path)


def test_backup_project_translates_stopped_container_to_project_error(
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
        project_service_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(
        ProjectError,
        match="Project 'demo' is not running\\. Start it first\\.",
    ):
        backup_project("demo", tmp_path)


def test_restore_project_does_not_precheck_cached_running_status(
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

    monkeypatch.setattr(project_service_module, "docker_swap_in_db", fake_swap_in_db)

    restore_project("demo", backup_file)

    assert captured["sql_bytes"] == b"select 1;\n"
    assert captured["container_name"] == "dr-llm-pg-demo"
    assert captured["target_db_name"] == "dr_llm"


def test_restore_project_translates_missing_container_to_project_not_found(
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

    monkeypatch.setattr(project_service_module, "docker_swap_in_db", fake_swap_in_db)

    with pytest.raises(ProjectNotFoundError, match="Project 'demo' not found"):
        restore_project("demo", backup_file)


def test_restore_project_translates_stopped_container_to_project_error(
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

    monkeypatch.setattr(project_service_module, "docker_swap_in_db", fake_swap_in_db)

    with pytest.raises(
        ProjectError,
        match="Project 'demo' is not running\\. Start it first\\.",
    ):
        restore_project("demo", backup_file)


def test_restore_project_missing_file_raises_native_file_not_found(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "missing.sql.gz"

    with pytest.raises(FileNotFoundError, match=str(backup_file)):
        restore_project("demo", backup_file)


def test_restore_project_rejects_plain_sql_files(
    tmp_path: Path,
) -> None:
    backup_file = tmp_path / "demo.sql"
    backup_file.write_text("select 1;\n")

    with pytest.raises(
        ProjectError,
        match=r"Restore only supports gzip-compressed SQL backups \(.sql.gz\)\.",
    ):
        restore_project("demo", backup_file)


def test_backup_project_removes_partial_file_when_streaming_fails(
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
        project_service_module,
        "call_docker_pg_dump_stream",
        fake_pg_dump_stream,
    )

    with pytest.raises(RuntimeError, match="pg_dump failed"):
        backup_project("demo", tmp_path)

    assert list((tmp_path / "demo").glob("*.sql.gz")) == []
