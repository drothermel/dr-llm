from __future__ import annotations

import gzip
import socket
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from dr_llm.project.docker_inspect import (
    get_all_docker_project_metadata,
    get_docker_project_metadata,
)
from dr_llm.project.docker_lifecycle import (
    call_docker_destroy,
    call_docker_start,
    call_docker_stop,
    create_project_container,
    wait_docker_ready,
)
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
    DockerProjectMetadata,
)
from dr_llm.project.docker_psql import (
    call_docker_pg_dump_stream,
    docker_swap_in_db,
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
from dr_llm.project.project_info import ProjectInfo

DOCKER_IMAGE = "postgres:16"
DEFAULT_BACKUP_DIR = Path.home() / ".dr-llm" / "backups"
BASE_PORT = 5500
PORT_PROBE_TIMEOUT_SECONDS = 0.05


def _port_has_listener(port: int) -> bool:
    with suppress(OSError):
        with socket.create_connection(
            ("127.0.0.1", port),
            timeout=PORT_PROBE_TIMEOUT_SECONDS,
        ):
            return True
    return False


def _find_available_port(
    claimed_ports: set[int],
    base: int = BASE_PORT,
    max_attempts: int = 100,
) -> int:
    for offset in range(max_attempts):
        port = base + offset
        if port not in claimed_ports:
            return port
    raise ProjectError(
        f"No available port found in range {base}-{base + max_attempts - 1}"
    )


def _project_from_metadata(metadata: DockerProjectMetadata) -> ProjectInfo:
    return ProjectInfo(
        name=metadata.name,
        port=metadata.port,
        status=metadata.status,
        created_at=metadata.created_at,
    )


def maybe_get_project(name: str) -> ProjectInfo | None:
    handle = ProjectInfo(name=name)
    metadata = get_docker_project_metadata(handle.container_name)
    if metadata is None:
        return None
    return _project_from_metadata(metadata)


def get_project(name: str) -> ProjectInfo:
    project = maybe_get_project(name)
    if project is None:
        raise ProjectNotFoundError(f"Project '{name}' not found")
    return project


def list_projects() -> list[ProjectInfo]:
    return [_project_from_metadata(m) for m in get_all_docker_project_metadata()]


def create_project(name: str) -> ProjectInfo:
    project = _allocate_port_and_create_container(name)
    project.status = _wait_ready_or_destroy(project)
    return project


def _allocate_port_and_create_container(name: str) -> ProjectInfo:
    claimed_ports = {
        metadata.port
        for metadata in get_all_docker_project_metadata()
        if metadata.port is not None
    }
    created_at = datetime.now(UTC)
    while True:
        port = _find_available_port(claimed_ports)
        if _port_has_listener(port):
            claimed_ports.add(port)
            continue
        project = ProjectInfo(name=name, port=port, created_at=created_at)
        try:
            create_project_container(
                volume_name=project.volume_name,
                container_name=project.container_name,
                db_name=ProjectInfo.db_name,
                db_user=ProjectInfo.db_user,
                db_password=ProjectInfo.db_password,
                docker_image=DOCKER_IMAGE,
                project=DockerProjectCreateMetadata(
                    name=project.name,
                    port=port,
                    created_at=created_at,
                ),
            )
        except DockerContainerConflictError as exc:
            raise ProjectAlreadyExistsError(f"Project '{name}' already exists") from exc
        except DockerPortAllocatedError:
            claimed_ports.add(port)
            continue
        return project


def _wait_ready_or_destroy(project: ProjectInfo) -> ContainerStatus:
    try:
        return wait_docker_ready(
            container_name=project.container_name,
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
        )
    except Exception as exc:
        _destroy_noting_cleanup_failure(project, original_exc=exc)
        raise


def _destroy_noting_cleanup_failure(
    project: ProjectInfo,
    *,
    original_exc: BaseException,
) -> None:
    try:
        call_docker_destroy(project.container_name, project.volume_name)
    except Exception as cleanup_exc:
        original_exc.add_note(
            f"Cleanup after failed project creation also failed: {cleanup_exc}"
        )


def start_project(name: str) -> ProjectInfo:
    handle = ProjectInfo(name=name)
    try:
        call_docker_start(handle.container_name)
        wait_docker_ready(
            container_name=handle.container_name,
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
        )
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc
    return get_project(name)


def stop_project(name: str) -> None:
    handle = ProjectInfo(name=name)
    try:
        call_docker_stop(handle.container_name)
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc


def destroy_project(name: str) -> None:
    handle = ProjectInfo(name=name)
    try:
        call_docker_destroy(handle.container_name, handle.volume_name)
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc


def backup_project(name: str, output_dir: Path | None = None) -> Path:
    handle = ProjectInfo(name=name)
    backup_dir = (output_dir or DEFAULT_BACKUP_DIR) / name
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"

    try:
        with gzip.open(backup_file, "wb") as backup_stream:
            try:
                call_docker_pg_dump_stream(
                    container_name=handle.container_name,
                    db_user=ProjectInfo.db_user,
                    db_name=ProjectInfo.db_name,
                    output_stream=backup_stream,  # type: ignore[arg-type]
                )
            except DockerContainerNotFoundError as exc:
                raise ProjectNotFoundError(f"Project '{name}' not found") from exc
            except DockerContainerNotRunningError as exc:
                raise ProjectError(
                    f"Project '{name}' is not running. Start it first."
                ) from exc
    except Exception:
        backup_file.unlink(missing_ok=True)
        raise

    return backup_file


def restore_project(name: str, backup_file: Path) -> None:
    if backup_file.suffixes[-2:] != [".sql", ".gz"]:
        raise ValueError("Restore only supports gzip-compressed SQL backups (.sql.gz).")

    handle = ProjectInfo(name=name)
    with gzip.open(backup_file, "rb") as sql_stream:
        try:
            docker_swap_in_db(
                sql_stream=sql_stream,  # type: ignore[arg-type]
                container_name=handle.container_name,
                db_user=ProjectInfo.db_user,
                target_db_name=ProjectInfo.db_name,
            )
        except DockerContainerNotFoundError as exc:
            raise ProjectNotFoundError(f"Project '{name}' not found") from exc
        except DockerContainerNotRunningError as exc:
            raise ProjectError(
                f"Project '{name}' is not running. Start it first."
            ) from exc
