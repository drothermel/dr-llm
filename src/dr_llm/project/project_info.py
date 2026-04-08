from __future__ import annotations

import gzip
import socket
from datetime import UTC, datetime
from pathlib import Path
from contextlib import suppress
from typing import IO, ClassVar, cast

from pydantic import BaseModel, computed_field

from dr_llm.project.docker import (
    call_docker_destroy,
    call_docker_pg_dump_stream,
    call_docker_start,
    call_docker_stop,
    create_project_container,
    docker_swap_in_db,
    get_all_docker_project_metadata,
    get_docker_project_metadata,
    wait_docker_ready,
)
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
    DockerProjectMetadata,
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


class ProjectInfo(BaseModel):
    db_name: ClassVar[str] = "dr_llm"
    db_user: ClassVar[str] = "postgres"
    db_password: ClassVar[str] = "postgres"
    docker_image: ClassVar[str] = "postgres:16"
    container_prefix: ClassVar[str] = "dr-llm-pg-"
    volume_prefix: ClassVar[str] = "dr-llm-data-"
    label_prefix: ClassVar[str] = "dr-llm.project"
    default_backup_dir: ClassVar[Path] = Path.home() / ".dr-llm" / "backups"

    name: str
    port: int | None = None
    status: ContainerStatus = ContainerStatus.UNKNOWN
    created_at: datetime | None = None

    @computed_field
    @property
    def dsn(self) -> str | None:
        if self.port is None:
            return None
        return self.get_dsn(self.port)

    @computed_field
    @property
    def volume_name(self) -> str:
        return self.get_volume_name(self.name)

    @computed_field
    @property
    def container_name(self) -> str:
        return self.get_container_name(self.name)

    @computed_field
    @property
    def running(self) -> bool:
        return self.status == ContainerStatus.RUNNING

    @classmethod
    def get_volume_name(cls, name: str) -> str:
        return f"{cls.volume_prefix}{name}"

    @classmethod
    def get_container_name(cls, name: str) -> str:
        return f"{cls.container_prefix}{name}"

    @classmethod
    def get_dsn(cls, port: int) -> str:
        return f"postgresql://{cls.db_user}:{cls.db_password}@localhost:{port}/{cls.db_name}"

    @classmethod
    def from_metadata(cls, metadata: DockerProjectMetadata) -> ProjectInfo:
        return ProjectInfo(
            name=metadata.name,
            port=metadata.port,
            status=metadata.status,
            created_at=metadata.created_at,
        )

    @classmethod
    def maybe_from_existing(cls, name: str) -> ProjectInfo | None:
        container_name = cls.get_container_name(name)
        metadata = get_docker_project_metadata(
            container_name,
            label_prefix=cls.label_prefix,
        )
        if metadata is None:
            return None
        return cls.from_metadata(metadata)

    @classmethod
    def get_by_name(cls, name: str) -> ProjectInfo:
        project_info = cls.maybe_from_existing(name)
        if project_info is None:
            raise ProjectNotFoundError(f"Project '{name}' not found")
        return project_info

    @classmethod
    def list_all(cls) -> list[ProjectInfo]:
        return [
            cls.from_metadata(metadata)
            for metadata in get_all_docker_project_metadata(cls.label_prefix)
        ]

    @classmethod
    def create_new(cls, name: str) -> ProjectInfo:
        claimed_ports = {
            metadata.port
            for metadata in get_all_docker_project_metadata(cls.label_prefix)
            if metadata.port is not None
        }
        created_at = datetime.now(UTC)

        while True:
            port = _find_available_port(claimed_ports)
            if _port_has_listener(port):
                claimed_ports.add(port)
                continue
            project_info = cls(
                name=name,
                port=port,
                created_at=created_at,
            )
            try:
                create_project_container(
                    volume_name=project_info.volume_name,
                    container_name=project_info.container_name,
                    db_name=project_info.db_name,
                    db_user=project_info.db_user,
                    db_password=project_info.db_password,
                    docker_image=project_info.docker_image,
                    project=DockerProjectCreateMetadata(
                        label_prefix=project_info.label_prefix,
                        name=project_info.name,
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

            try:
                project_info.status = wait_docker_ready(
                    container_name=project_info.container_name,
                    db_user=ProjectInfo.db_user,
                    db_name=ProjectInfo.db_name,
                )
            except Exception as exc:
                try:
                    call_docker_destroy(
                        project_info.container_name,
                        project_info.volume_name,
                    )
                except Exception as cleanup_exc:
                    exc.add_note(
                        "Cleanup after failed project creation also failed: "
                        f"{cleanup_exc}"
                    )
                raise
            return project_info

    @classmethod
    def start(cls, name: str) -> ProjectInfo:
        try:
            call_docker_start(cls.get_container_name(name))
            wait_docker_ready(
                container_name=cls.get_container_name(name),
                db_user=cls.db_user,
                db_name=cls.db_name,
            )
        except DockerContainerNotFoundError as exc:
            raise ProjectNotFoundError(f"Project '{name}' not found") from exc
        return cls.get_by_name(name)

    @classmethod
    def stop(cls, name: str) -> None:
        try:
            call_docker_stop(cls.get_container_name(name))
        except DockerContainerNotFoundError as exc:
            raise ProjectNotFoundError(f"Project '{name}' not found") from exc

    @classmethod
    def destroy(cls, name: str) -> None:
        try:
            call_docker_destroy(cls.get_container_name(name), cls.get_volume_name(name))
        except DockerContainerNotFoundError as exc:
            raise ProjectNotFoundError(f"Project '{name}' not found") from exc

    @classmethod
    def backup(cls, name: str, output_dir: Path | None = None) -> Path:
        backup_dir = (output_dir or cls.default_backup_dir) / name
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"

        try:
            with gzip.open(backup_file, "wb") as backup_stream:
                try:
                    call_docker_pg_dump_stream(
                        container_name=cls.get_container_name(name),
                        db_user=cls.db_user,
                        db_name=cls.db_name,
                        output_stream=cast(IO[bytes], backup_stream),
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

    @classmethod
    def restore(cls, name: str, backup_file: Path) -> None:
        if backup_file.suffixes[-2:] != [".sql", ".gz"]:
            raise ValueError(
                "Restore only supports gzip-compressed SQL backups (.sql.gz)."
            )

        with gzip.open(backup_file, "rb") as sql_stream:
            try:
                docker_swap_in_db(
                    sql_stream=cast(IO[bytes], sql_stream),
                    container_name=cls.get_container_name(name),
                    db_user=cls.db_user,
                    target_db_name=cls.db_name,
                )
            except DockerContainerNotFoundError as exc:
                raise ProjectNotFoundError(f"Project '{name}' not found") from exc
            except DockerContainerNotRunningError as exc:
                raise ProjectError(
                    f"Project '{name}' is not running. Start it first."
                ) from exc
