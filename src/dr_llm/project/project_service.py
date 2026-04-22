from __future__ import annotations

import gzip
import logging
import socket
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path

from dr_llm.datetime_utils import UTC, normalize_utc
from dr_llm.errors import TransientPersistenceError
from dr_llm.project.models import (
    CreateProjectRequest,
    ProjectCreationBlockReason,
    ProjectCreationReadiness,
    ProjectCreationViolation,
    ProjectInspectionSummary,
    ProjectPoolInspection,
    ProjectPoolInspectionReason,
    ProjectPoolInspectionStatus,
)
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
    wait_dsn_ready,
)
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
    DockerProjectMetadata,
)
from dr_llm.project.docker_psql import (
    call_docker_pg_dump_stream,
    docker_swap_in_db,
    validate_pg_identifier,
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

validate_pg_identifier(ProjectInfo.db_name, "database name")
logger = logging.getLogger(__name__)


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
    metadata = get_docker_project_metadata(ProjectInfo.container_name_for(name))
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


def inspect_projects() -> list[ProjectInspectionSummary]:
    return [
        ProjectInspectionSummary(
            project=project,
            pool_inspection=_inspect_project_pools(project),
        )
        for project in list_projects()
    ]


def assess_project_creation(
    request: CreateProjectRequest,
    *,
    cooldown_seconds: int = 60,
) -> ProjectCreationReadiness:
    projects = list_projects()
    violations: list[ProjectCreationViolation] = []
    if not request.name_is_valid:
        violations.append(
            ProjectCreationViolation(
                reason=ProjectCreationBlockReason.invalid_name,
                message=(
                    "project_name must be lowercase alphanumeric with underscores, "
                    f"starting with a letter; got {request.project_name!r}"
                ),
                project_name=request.project_name,
            )
        )
    if any(project.name == request.project_name for project in projects):
        violations.append(
            ProjectCreationViolation(
                reason=ProjectCreationBlockReason.already_exists,
                message=f"Project {request.project_name!r} already exists.",
                project_name=request.project_name,
            )
        )

    cutoff = datetime.now(UTC) - timedelta(seconds=cooldown_seconds)
    recent_project_names = [
        project.name
        for project in projects
        if (created_at := normalize_utc(project.created_at)) is not None
        and created_at >= cutoff
    ]
    if recent_project_names:
        violations.append(
            ProjectCreationViolation(
                reason=ProjectCreationBlockReason.cooldown_active,
                message=(
                    "Cannot create a new project yet; recent projects are still within "
                    "the cooldown window: " + ", ".join(recent_project_names)
                ),
            )
        )

    return ProjectCreationReadiness(
        request=request,
        existing_projects=projects,
        recent_project_names=recent_project_names,
        violations=violations,
    )


def create_project(request: CreateProjectRequest) -> ProjectInfo:
    readiness = assess_project_creation(request)
    if not readiness.allowed:
        duplicate = next(
            (
                violation
                for violation in readiness.violations
                if violation.reason == ProjectCreationBlockReason.already_exists
            ),
            None,
        )
        if duplicate is not None:
            raise ProjectAlreadyExistsError(duplicate.message)
        assert readiness.blocked_message is not None
        raise ProjectError(readiness.blocked_message)

    name = request.project_name
    claimed_ports = _collect_claimed_ports()
    project = _create_container_with_port_retry(name, claimed_ports)
    status = _wait_ready_or_destroy(project)
    return project.model_copy(update={"status": status})


def _collect_claimed_ports() -> set[int]:
    return {
        metadata.port
        for metadata in get_all_docker_project_metadata()
        if metadata.port is not None
    }


def _inspect_project_pools(project: ProjectInfo) -> ProjectPoolInspection:
    from dr_llm.pool.admin_service import discover_pools

    inspected_at = datetime.now(UTC)
    if not project.running:
        return ProjectPoolInspection(
            status=ProjectPoolInspectionStatus.skipped,
            reason=ProjectPoolInspectionReason.project_not_running,
            message="Project is not running.",
            inspected_at=inspected_at,
        )
    if project.dsn is None:
        return ProjectPoolInspection(
            status=ProjectPoolInspectionStatus.skipped,
            reason=ProjectPoolInspectionReason.missing_dsn,
            message="Project has no DSN.",
            inspected_at=inspected_at,
        )
    try:
        return ProjectPoolInspection(
            status=ProjectPoolInspectionStatus.discovered,
            pool_names=discover_pools(project.dsn),
            inspected_at=inspected_at,
        )
    except TransientPersistenceError:
        logger.warning(
            "Could not connect to project database during pool inspection project=%r",
            project.name,
        )
        return ProjectPoolInspection(
            status=ProjectPoolInspectionStatus.failed,
            reason=ProjectPoolInspectionReason.connection_failed,
            message="Could not connect to project database.",
            inspected_at=inspected_at,
        )
    except Exception:
        logger.exception(
            "Unexpected pool inspection failure for project %r",
            project.name,
        )
        return ProjectPoolInspection(
            status=ProjectPoolInspectionStatus.failed,
            reason=ProjectPoolInspectionReason.unexpected_error,
            message="Pool inspection failed.",
            inspected_at=inspected_at,
        )


def _create_container_with_port_retry(
    name: str, claimed_ports: set[int]
) -> ProjectInfo:
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
        status = wait_docker_ready(
            container_name=project.container_name,
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
        )
        # Defeat the startup race where pg_isready returns OK against the
        # in-container Unix socket while the host-mapped TCP listener is
        # still bouncing connections during init.
        assert project.dsn is not None, "newly-created project must have a port"
        wait_dsn_ready(project.dsn)
        return status
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
    container_name = ProjectInfo.container_name_for(name)
    try:
        call_docker_start(container_name)
        wait_docker_ready(
            container_name=container_name,
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
        )
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc
    project = get_project(name)
    if project.dsn is not None:
        wait_dsn_ready(project.dsn)
    return project


def stop_project(name: str) -> None:
    try:
        call_docker_stop(ProjectInfo.container_name_for(name))
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc


def destroy_project(name: str) -> None:
    try:
        call_docker_destroy(
            ProjectInfo.container_name_for(name),
            ProjectInfo.volume_name_for(name),
        )
    except DockerContainerNotFoundError as exc:
        raise ProjectNotFoundError(f"Project '{name}' not found") from exc


def backup_project(name: str, output_dir: Path | None = None) -> Path:
    backup_dir = (output_dir or DEFAULT_BACKUP_DIR) / name
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"

    try:
        with gzip.open(backup_file, "wb") as backup_stream:
            try:
                call_docker_pg_dump_stream(
                    container_name=ProjectInfo.container_name_for(name),
                    db_user=ProjectInfo.db_user,
                    db_name=ProjectInfo.db_name,
                    output_stream=backup_stream,
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
        raise ProjectError(
            "Restore only supports gzip-compressed SQL backups (.sql.gz)."
        )

    with gzip.open(backup_file, "rb") as sql_stream:
        try:
            docker_swap_in_db(
                sql_stream=sql_stream,
                container_name=ProjectInfo.container_name_for(name),
                db_user=ProjectInfo.db_user,
                target_db_name=ProjectInfo.db_name,
            )
        except DockerContainerNotFoundError as exc:
            raise ProjectNotFoundError(f"Project '{name}' not found") from exc
        except DockerContainerNotRunningError as exc:
            raise ProjectError(
                f"Project '{name}' is not running. Start it first."
            ) from exc
