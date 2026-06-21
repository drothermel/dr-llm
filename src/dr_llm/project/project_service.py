from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
import gzip
import logging
import socket
import subprocess
import tempfile
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
import re
from time import perf_counter
from typing import Final, IO, cast
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from dr_llm.datetime_utils import UTC, normalize_utc
from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.admin.deletion import (
    DeletePoolRequest,
    PoolDeletionResult,
    PoolDeletionStatus,
)
from dr_llm.project.models import (
    CreateProjectRequest,
    DeleteProjectRequest,
    ProjectCreationBlockReason,
    ProjectCreationReadiness,
    ProjectCreationViolation,
    ProjectDeletionBlockReason,
    ProjectDeletionReadiness,
    ProjectDeletionResult,
    ProjectDeletionStatus,
    ProjectDeletionViolation,
    ProjectInspectionSummary,
    ProjectNeonSyncResult,
    ProjectPoolInspection,
    ProjectPoolInspectionReason,
    ProjectPoolInspectionStatus,
    ProjectSyncValidation,
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
from dr_llm.project.docker_runner import BinaryStream
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
_POOL_DELETE_MAX_WORKERS: Final[int] = 2
_BACKUP_PROGRESS_ROWS_STEP: Final[int] = 20_000
_COPY_FROM_STDIN_RE: Final[re.Pattern[bytes]] = re.compile(
    rb'^COPY\s+(?P<table>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?)\s+\(.*\)\s+FROM stdin;$'
)

validate_pg_identifier(ProjectInfo.db_name, "database name")
logger = logging.getLogger(__name__)


class _BackupProgressWriter:
    def __init__(
        self,
        *,
        project_name: str,
        destination: Path,
        sink: BinaryStream,
    ) -> None:
        self.project_name = project_name
        self.destination = destination
        self._sink = sink
        self._tracker = _SqlCopyProgressTracker(
            project_name=project_name,
            operation="backup",
            action_verb="dumping",
            completion_verb="dumped",
        )

    @property
    def bytes_processed(self) -> int:
        return self._tracker.bytes_processed

    @property
    def rows_processed(self) -> int:
        return self._tracker.rows_processed

    def write(self, data: bytes) -> int:
        written = self._sink.write(data)
        self._tracker.process_bytes(data)
        return written

    def flush(self) -> None:
        self._sink.flush()

    def finalize(self) -> None:
        self._tracker.finalize()


class _RestoreProgressReader:
    def __init__(
        self,
        *,
        project_name: str,
        source: BinaryStream,
    ) -> None:
        self.project_name = project_name
        self._source = source
        self._tracker = _SqlCopyProgressTracker(
            project_name=project_name,
            operation="restore",
            action_verb="restoring",
            completion_verb="restored",
        )

    @property
    def bytes_processed(self) -> int:
        return self._tracker.bytes_processed

    @property
    def rows_processed(self) -> int:
        return self._tracker.rows_processed

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)
        if data:
            self._tracker.process_bytes(data)
        return data

    def close(self) -> None:
        self._source.close()

    def finalize(self) -> None:
        self._tracker.finalize()


class _SqlCopyProgressTracker:
    def __init__(
        self,
        *,
        project_name: str,
        operation: str,
        action_verb: str,
        completion_verb: str,
    ) -> None:
        self.project_name = project_name
        self.operation = operation
        self.action_verb = action_verb
        self.completion_verb = completion_verb
        self._bytes_processed = 0
        self._rows_processed = 0
        self._next_rows_log = _BACKUP_PROGRESS_ROWS_STEP
        self._line_buffer = bytearray()
        self._active_copy_table: str | None = None
        self._active_copy_rows = 0

    @property
    def bytes_processed(self) -> int:
        return self._bytes_processed

    @property
    def rows_processed(self) -> int:
        return self._rows_processed

    def process_bytes(self, data: bytes) -> None:
        self._bytes_processed += len(data)
        self._track_dump_lines(data)

    def finalize(self) -> None:
        if self._line_buffer:
            self._process_dump_line(bytes(self._line_buffer))
            self._line_buffer.clear()
        if self._active_copy_table is not None:
            self._log_completed_copy_table()

    def _track_dump_lines(self, data: bytes) -> None:
        self._line_buffer.extend(data)
        while True:
            newline_index = self._line_buffer.find(b"\n")
            if newline_index < 0:
                return
            line = bytes(self._line_buffer[:newline_index])
            del self._line_buffer[: newline_index + 1]
            self._process_dump_line(line.rstrip(b"\r"))

    def _process_dump_line(self, line: bytes) -> None:
        if self._active_copy_table is None:
            table_name = _parse_copy_table_name(line)
            if table_name is None:
                return
            self._active_copy_table = table_name
            self._active_copy_rows = 0
            logger.info(
                "%s for project %r started %s table %s",
                self.operation.capitalize(),
                self.project_name,
                self.action_verb,
                table_name,
            )
            return

        if line == b"\\.":
            self._log_completed_copy_table()
            self._active_copy_table = None
            self._active_copy_rows = 0
            return

        self._rows_processed += 1
        self._active_copy_rows += 1
        self._log_row_progress()

    def _log_row_progress(self) -> None:
        while self._rows_processed >= self._next_rows_log:
            logger.info(
                "%s for project %r progress: processed %s rows",
                self.operation.capitalize(),
                self.project_name,
                _format_count(self._next_rows_log),
            )
            self._next_rows_log += _BACKUP_PROGRESS_ROWS_STEP

    def _log_completed_copy_table(self) -> None:
        assert self._active_copy_table is not None
        logger.info(
            "%s for project %r %s %s rows from table %s",
            self.operation.capitalize(),
            self.project_name,
            self.completion_verb,
            _format_count(self._active_copy_rows),
            self._active_copy_table,
        )


def _parse_copy_table_name(line: bytes) -> str | None:
    match = _COPY_FROM_STDIN_RE.match(line)
    if match is None:
        return None
    return match.group("table").decode("utf-8")


def _format_count(value: int) -> str:
    return f"{value:,}"


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
    metadata = get_docker_project_metadata(
        ProjectInfo.container_name_for(name)
    )
    if metadata is None:
        return None
    return _project_from_metadata(metadata)


def get_project(name: str) -> ProjectInfo:
    project = maybe_get_project(name)
    if project is None:
        raise ProjectNotFoundError(f"Project '{name}' not found")
    return project


def list_projects() -> list[ProjectInfo]:
    return [
        _project_from_metadata(m) for m in get_all_docker_project_metadata()
    ]


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
                if violation.reason
                == ProjectCreationBlockReason.already_exists
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


def assess_project_deletion(
    request: DeleteProjectRequest,
) -> ProjectDeletionReadiness:
    project = maybe_get_project(request.project_name)
    if project is None:
        return ProjectDeletionReadiness(
            request=request,
            violations=[
                ProjectDeletionViolation(
                    reason=ProjectDeletionBlockReason.project_not_found,
                    message=f"Project {request.project_name!r} not found",
                    project_name=request.project_name,
                )
            ],
        )
    return ProjectDeletionReadiness(request=request, project=project)


def delete_project(request: DeleteProjectRequest) -> ProjectDeletionResult:
    readiness = assess_project_deletion(request)
    if not readiness.allowed:
        return ProjectDeletionResult(
            request=request,
            project=readiness.project,
            status=ProjectDeletionStatus.blocked,
            violations=readiness.violations,
            message=readiness.blocked_message,
        )

    assert readiness.project is not None
    project = readiness.project
    temporarily_started = False
    running_project = project
    discovered_pool_names: list[str] = []
    pool_results: list[PoolDeletionResult] = []
    destroyed_project_resources = False
    failure_message: str | None = None
    violations: list[ProjectDeletionViolation] = []

    try:
        if not project.running:
            running_project = start_project(request.project_name)
            temporarily_started = True
        if running_project.dsn is None:
            violations.append(
                ProjectDeletionViolation(
                    reason=ProjectDeletionBlockReason.project_missing_dsn,
                    message=f"Project {running_project.name!r} has no DSN.",
                    project_name=running_project.name,
                )
            )
            failure_message = violations[0].message
            return ProjectDeletionResult(
                request=request,
                project=running_project,
                status=ProjectDeletionStatus.blocked,
                discovered_pool_names=discovered_pool_names,
                pool_results=pool_results,
                temporarily_started=temporarily_started,
                destroyed_project_resources=False,
                violations=violations,
                message=failure_message,
            )

        from dr_llm.pool.admin.discovery import discover_pools

        discovered_pool_names = discover_pools(running_project.dsn)
        pool_results = _delete_pools_for_project(
            request.project_name,
            discovered_pool_names,
        )
        if any(
            result.status != PoolDeletionStatus.deleted
            for result in pool_results
        ):
            failure_message = _project_delete_failure_message(pool_results)
        else:
            call_docker_destroy(
                running_project.container_name,
                running_project.volume_name,
            )
            destroyed_project_resources = True
            return ProjectDeletionResult(
                request=request,
                project=running_project,
                status=ProjectDeletionStatus.deleted,
                discovered_pool_names=discovered_pool_names,
                pool_results=pool_results,
                temporarily_started=temporarily_started,
                destroyed_project_resources=True,
                message=(
                    f"Project {request.project_name!r} and all discovered pools were deleted."
                ),
            )
    except Exception as exc:
        failure_message = str(exc)

    stop_failure_message: str | None = None
    if temporarily_started and not destroyed_project_resources:
        try:
            stop_project(request.project_name)
        except Exception as exc:
            stop_failure_message = str(exc)

    if stop_failure_message is not None:
        failure_message = (
            f"{failure_message or 'Project deletion failed.'} "
            f"Failed to restore the original stopped state: {stop_failure_message}"
        )

    return ProjectDeletionResult(
        request=request,
        project=running_project,
        status=ProjectDeletionStatus.failed,
        discovered_pool_names=discovered_pool_names,
        pool_results=pool_results,
        temporarily_started=temporarily_started,
        destroyed_project_resources=False,
        message=failure_message or "Project deletion failed.",
    )


def _collect_claimed_ports() -> set[int]:
    return {
        metadata.port
        for metadata in get_all_docker_project_metadata()
        if metadata.port is not None
    }


def _inspect_project_pools(project: ProjectInfo) -> ProjectPoolInspection:
    from dr_llm.pool.admin.discovery import discover_pools

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
            raise ProjectAlreadyExistsError(
                f"Project '{name}' already exists"
            ) from exc
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
        assert project.dsn is not None, (
            "newly-created project must have a port"
        )
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
    result = delete_project(DeleteProjectRequest(project_name=name))
    if result.status == ProjectDeletionStatus.deleted:
        return
    violation = next(
        (
            violation
            for violation in result.violations
            if violation.reason == ProjectDeletionBlockReason.project_not_found
        ),
        None,
    )
    if violation is not None:
        raise ProjectNotFoundError(violation.message)
    raise ProjectError(result.message or "Project deletion failed.")


def backup_project(
    name: str, output_dir: Path | None = None, *, portable: bool = False
) -> Path:
    backup_dir = (output_dir or DEFAULT_BACKUP_DIR) / name
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"
    started_at = perf_counter()

    logger.info("Starting backup for project %r to %s", name, backup_file)
    progress_stream: _BackupProgressWriter | None = None

    try:
        with gzip.open(backup_file, "wb") as backup_stream:
            progress_stream = _BackupProgressWriter(
                project_name=name,
                destination=backup_file,
                sink=backup_stream,
            )
            try:
                call_docker_pg_dump_stream(
                    container_name=ProjectInfo.container_name_for(name),
                    db_user=ProjectInfo.db_user,
                    db_name=ProjectInfo.db_name,
                    output_stream=cast(IO[bytes], progress_stream),
                    extra_args=_pg_dump_portable_args(portable),
                )
                progress_stream.finalize()
            except DockerContainerNotFoundError as exc:
                raise ProjectNotFoundError(
                    f"Project '{name}' not found"
                ) from exc
            except DockerContainerNotRunningError as exc:
                raise ProjectError(
                    f"Project '{name}' is not running. Start it first."
                ) from exc
    except Exception:
        logger.exception(
            "Backup for project %r failed after processing %s dump bytes and %s rows",
            name,
            _format_count(progress_stream.bytes_processed)
            if progress_stream is not None
            else "0",
            _format_count(progress_stream.rows_processed)
            if progress_stream is not None
            else "0",
        )
        backup_file.unlink(missing_ok=True)
        raise

    elapsed_seconds = perf_counter() - started_at
    logger.info(
        "Completed backup for project %r to %s in %.2fs (%s dump bytes, %s rows, %s compressed bytes)",
        name,
        backup_file,
        elapsed_seconds,
        _format_count(progress_stream.bytes_processed),
        _format_count(progress_stream.rows_processed),
        _format_count(backup_file.stat().st_size),
    )

    return backup_file


def _pg_dump_portable_args(portable: bool) -> tuple[str, ...]:
    if not portable:
        return ()
    return ("--no-owner", "--no-privileges")


def restore_project(name: str, backup_file: Path) -> None:
    if backup_file.suffixes[-2:] != [".sql", ".gz"]:
        raise ProjectError(
            "Restore only supports gzip-compressed SQL backups (.sql.gz)."
        )

    started_at = perf_counter()
    logger.info("Starting restore for project %r from %s", name, backup_file)
    progress_stream: _RestoreProgressReader | None = None

    try:
        with gzip.open(backup_file, "rb") as sql_stream:
            progress_stream = _RestoreProgressReader(
                project_name=name,
                source=sql_stream,
            )
            try:
                docker_swap_in_db(
                    sql_stream=cast(IO[bytes], progress_stream),
                    container_name=ProjectInfo.container_name_for(name),
                    db_user=ProjectInfo.db_user,
                    target_db_name=ProjectInfo.db_name,
                )
                progress_stream.finalize()
            except DockerContainerNotFoundError as exc:
                raise ProjectNotFoundError(
                    f"Project '{name}' not found"
                ) from exc
            except DockerContainerNotRunningError as exc:
                raise ProjectError(
                    f"Project '{name}' is not running. Start it first."
                ) from exc
    except Exception:
        logger.exception(
            "Restore for project %r failed after processing %s SQL bytes and %s rows",
            name,
            _format_count(progress_stream.bytes_processed)
            if progress_stream is not None
            else "0",
            _format_count(progress_stream.rows_processed)
            if progress_stream is not None
            else "0",
        )
        raise

    elapsed_seconds = perf_counter() - started_at
    logger.info(
        "Completed restore for project %r from %s in %.2fs (%s SQL bytes, %s rows, %s compressed bytes)",
        name,
        backup_file,
        elapsed_seconds,
        _format_count(progress_stream.bytes_processed),
        _format_count(progress_stream.rows_processed),
        _format_count(backup_file.stat().st_size),
    )


def sync_project_to_neon(
    name: str,
    admin_url: str,
    *,
    target_database: str | None = None,
    drop_previous: bool = False,
) -> ProjectNeonSyncResult:
    """Replace a Neon database with a validated snapshot of a local project."""

    project = get_project(name)
    if project.dsn is None:
        raise ProjectError(f"Project {name!r} has no DSN; start it first.")

    target_name = validate_pg_identifier(
        target_database or name, "database name"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    temporary_database = _sync_database_name(target_name, "sync", timestamp)
    previous_database = _sync_database_name(target_name, "prev", timestamp)

    started_at = perf_counter()
    logger.info(
        "Starting Neon sync for project %r to database %r",
        name,
        target_name,
    )
    _create_database(admin_url, temporary_database)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"dr_llm_{name}_",
            suffix=".sql",
            delete=True,
        ) as dump_file:
            _dump_project_to_file(name, Path(dump_file.name), portable=True)
            _restore_sql_file(
                _url_for_database(admin_url, temporary_database),
                Path(dump_file.name),
            )
        validation = validate_project_database_copy(
            source_dsn=project.dsn,
            target_dsn=_url_for_database(admin_url, temporary_database),
        )
        if not validation.passed:
            raise ProjectError(
                "Neon sync validation failed before swap: "
                + "; ".join(validation.mismatches)
            )
        replaced_existing = _replace_database(
            admin_url=admin_url,
            temporary_database=temporary_database,
            target_database=target_name,
            previous_database=previous_database,
        )
        if drop_previous and replaced_existing:
            _drop_database(admin_url, previous_database)
        logger.info(
            "Completed Neon sync for project %r to database %r in %.2fs",
            name,
            target_name,
            perf_counter() - started_at,
        )
        return ProjectNeonSyncResult(
            project_name=name,
            target_database=target_name,
            temporary_database=temporary_database,
            previous_database=previous_database if replaced_existing else None,
            previous_database_dropped=drop_previous and replaced_existing,
            validation=validation,
        )
    except Exception:
        with suppress(Exception):
            _drop_database(admin_url, temporary_database)
        raise


def validate_project_database_copy(
    *,
    source_dsn: str,
    target_dsn: str,
) -> ProjectSyncValidation:
    source_tables = _public_table_names(source_dsn)
    target_tables = _public_table_names(target_dsn)
    mismatches: list[str] = []
    if source_tables != target_tables:
        missing = sorted(set(source_tables) - set(target_tables))
        extra = sorted(set(target_tables) - set(source_tables))
        if missing:
            mismatches.append("missing target tables: " + ", ".join(missing))
        if extra:
            mismatches.append("extra target tables: " + ", ".join(extra))

    source_pool_count = _pool_catalog_count(source_dsn)
    target_pool_count = _pool_catalog_count(target_dsn)
    if source_pool_count != target_pool_count:
        mismatches.append(
            "pool_catalog count mismatch: "
            f"source={source_pool_count} target={target_pool_count}"
        )

    checked_table_count = 0
    if source_tables == target_tables:
        for table_name in source_tables:
            source_rows = _table_row_count(source_dsn, table_name)
            target_rows = _table_row_count(target_dsn, table_name)
            checked_table_count += 1
            if source_rows != target_rows:
                mismatches.append(
                    f"{table_name} row count mismatch: "
                    f"source={source_rows} target={target_rows}"
                )

    return ProjectSyncValidation(
        source_table_count=len(source_tables),
        target_table_count=len(target_tables),
        source_pool_count=source_pool_count,
        target_pool_count=target_pool_count,
        checked_table_count=checked_table_count,
        mismatches=mismatches,
    )


def _sync_database_name(
    target_database: str, label: str, timestamp: str
) -> str:
    suffix = f"_{label}_{timestamp}"
    max_prefix_length = 63 - len(suffix)
    return validate_pg_identifier(
        f"{target_database[:max_prefix_length]}{suffix}", "database name"
    )


def _url_for_database(base_url: str, database_name: str) -> str:
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ProjectError("Database URL must include a scheme and host.")
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"/{database_name}",
            parts.query,
            parts.fragment,
        )
    )


def _dump_project_to_file(
    name: str, dump_path: Path, *, portable: bool
) -> None:
    with dump_path.open("wb") as output_stream:
        call_docker_pg_dump_stream(
            container_name=ProjectInfo.container_name_for(name),
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
            output_stream=output_stream,
            extra_args=_pg_dump_portable_args(portable),
        )


def _restore_sql_file(target_dsn: str, dump_path: Path) -> None:
    try:
        with dump_path.open("rb") as sql_stream:
            result = subprocess.run(
                ["psql", target_dsn, "-v", "ON_ERROR_STOP=1", "-q"],
                stdin=sql_stream,
                capture_output=True,
                check=False,
            )
    except FileNotFoundError as exc:
        raise ProjectError(
            "psql is required to restore remote databases."
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise ProjectError(f"psql restore failed: {stderr or 'unknown error'}")


def _connect_admin(admin_url: str) -> psycopg.Connection[tuple]:
    try:
        return psycopg.connect(admin_url, autocommit=True)
    except psycopg.Error as exc:
        raise ProjectError(
            f"Could not connect to remote Postgres: {exc}"
        ) from exc


def _create_database(admin_url: str, database_name: str) -> None:
    with _connect_admin(admin_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(database_name)
                )
            )


def _drop_database(admin_url: str, database_name: str) -> None:
    with _connect_admin(admin_url) as conn:
        with conn.cursor() as cur:
            _terminate_database_connections(cur, database_name)
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(database_name)
                )
            )


def _replace_database(
    *,
    admin_url: str,
    temporary_database: str,
    target_database: str,
    previous_database: str,
) -> bool:
    with _connect_admin(admin_url) as conn:
        with conn.cursor() as cur:
            target_exists = _database_exists(cur, target_database)
            if target_exists:
                _terminate_database_connections(cur, target_database)
                cur.execute(
                    sql.SQL("ALTER DATABASE {} RENAME TO {}").format(
                        sql.Identifier(target_database),
                        sql.Identifier(previous_database),
                    )
                )
            cur.execute(
                sql.SQL("ALTER DATABASE {} RENAME TO {}").format(
                    sql.Identifier(temporary_database),
                    sql.Identifier(target_database),
                )
            )
    return target_exists


def _database_exists(
    cur: psycopg.Cursor[tuple],
    database_name: str,
) -> bool:
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        [database_name],
    )
    return cur.fetchone() is not None


def _terminate_database_connections(
    cur: psycopg.Cursor[tuple],
    database_name: str,
) -> None:
    cur.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s
          AND pid <> pg_backend_pid()
        """,
        [database_name],
    )


def _public_table_names(dsn: str) -> list[str]:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relname
                FROM pg_class
                WHERE relkind = 'r'
                  AND relnamespace = 'public'::regnamespace
                ORDER BY relname
                """
            )
            return [row[0] for row in cur.fetchall()]


def _pool_catalog_count(dsn: str) -> int | None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.pool_catalog') IS NOT NULL"
            )
            exists_row = cur.fetchone()
            assert exists_row is not None
            if not exists_row[0]:
                return None
            cur.execute("SELECT count(*) FROM pool_catalog")
            count_row = cur.fetchone()
            assert count_row is not None
            return count_row[0]


def _table_row_count(dsn: str, table_name: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}").format(
                    sql.Identifier(table_name)
                )
            )
            count_row = cur.fetchone()
            assert count_row is not None
            return count_row[0]


def _delete_pools_for_project(
    project_name: str,
    pool_names: list[str],
) -> list[PoolDeletionResult]:
    if not pool_names:
        return []

    max_workers = max(1, min(_POOL_DELETE_MAX_WORKERS, len(pool_names)))
    submitted_index = 0
    failure_seen = False
    results_by_index: dict[int, PoolDeletionResult] = {}
    futures: dict[Future[PoolDeletionResult], int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while submitted_index < max_workers:
            future = executor.submit(
                _delete_single_pool_for_project,
                project_name,
                pool_names[submitted_index],
            )
            futures[future] = submitted_index
            submitted_index += 1

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                result = future.result()
                results_by_index[index] = result
                if result.status != PoolDeletionStatus.deleted:
                    failure_seen = True

            if failure_seen:
                while submitted_index < len(pool_names):
                    pool_name = pool_names[submitted_index]
                    results_by_index[submitted_index] = PoolDeletionResult(
                        request=DeletePoolRequest(
                            project_name=project_name,
                            pool_name=pool_name,
                        ),
                        status=PoolDeletionStatus.cancelled,
                        message=(
                            "Pool deletion was not started because an earlier pool "
                            "deletion failed."
                        ),
                    )
                    submitted_index += 1
                continue

            while (
                submitted_index < len(pool_names)
                and len(futures) < max_workers
            ):
                future = executor.submit(
                    _delete_single_pool_for_project,
                    project_name,
                    pool_names[submitted_index],
                )
                futures[future] = submitted_index
                submitted_index += 1

    return [results_by_index[index] for index in range(len(pool_names))]


def _delete_single_pool_for_project(
    project_name: str,
    pool_name: str,
) -> PoolDeletionResult:
    from dr_llm.pool.admin.deletion import delete_pool

    return delete_pool(
        DeletePoolRequest(project_name=project_name, pool_name=pool_name)
    )


def _project_delete_failure_message(
    pool_results: list[PoolDeletionResult],
) -> str:
    first_failure = next(
        (
            result
            for result in pool_results
            if result.status != PoolDeletionStatus.deleted
        ),
        None,
    )
    if first_failure is None:
        return "Project deletion failed."
    return (
        f"Project deletion stopped after pool {first_failure.request.pool_name!r} "
        f"reported status {first_failure.status.value}: "
        f"{first_failure.message or 'no additional detail'}"
    )
