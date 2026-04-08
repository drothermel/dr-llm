from __future__ import annotations

import gzip
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, computed_field

from dr_llm.project.docker import (
    ContainerStatus,
    DockerProjectMetadata,
    call_docker_create,
    call_docker_destroy,
    call_docker_pg_dump,
    call_docker_start,
    call_docker_stop,
    docker_swap_in_db,
    get_all_docker_project_metadata,
    get_claimed_project_ports,
    get_docker_project_metadata,
    wait_docker_ready,
)

BASE_PORT = 5500


def _port_is_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _find_available_port(
    claimed_ports: set[int],
    base: int = BASE_PORT,
    max_attempts: int = 100,
) -> int:
    for offset in range(max_attempts):
        port = base + offset
        if port not in claimed_ports and _port_is_free(port):
            return port
    raise RuntimeError(
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
    created_at: str | None = None

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
    def check_exists(cls, container_name: str) -> bool:
        return (
            get_docker_project_metadata(
                container_name,
                label_prefix=cls.label_prefix,
            )
            is not None
        )

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
            raise RuntimeError(f"Project '{name}' not found")
        return project_info

    @classmethod
    def list_all(cls) -> list[ProjectInfo]:
        return [
            cls.from_metadata(metadata)
            for metadata in get_all_docker_project_metadata(cls.label_prefix)
        ]

    @classmethod
    def create_new(cls, name: str) -> ProjectInfo:
        claimed_ports = get_claimed_project_ports(cls.label_prefix)
        project_info = cls(
            name=name,
            port=_find_available_port(claimed_ports),
            created_at=datetime.now(UTC).isoformat(),
        )
        project_info.verify_not_exists()
        call_docker_create(
            volume_name=project_info.volume_name,
            container_name=project_info.container_name,
            db_name=project_info.db_name,
            db_user=project_info.db_user,
            db_password=project_info.db_password,
            label_prefix=project_info.label_prefix,
            name=project_info.name,
            port=project_info.port,
            created_at=project_info.created_at,
            docker_image=project_info.docker_image,
        )
        project_info.status = wait_docker_ready(
            container_name=project_info.container_name,
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
        )
        return project_info

    def verify_exists(self) -> None:
        if not ProjectInfo.check_exists(self.container_name):
            raise RuntimeError(f"Project '{self.name}' not found")

    def verify_not_exists(self) -> None:
        if ProjectInfo.check_exists(self.container_name):
            raise RuntimeError(f"Project '{self.name}' already exists")

    def start(self) -> None:
        self.verify_exists()
        if self.status != ContainerStatus.RUNNING:
            call_docker_start(self.container_name)
            self.status = wait_docker_ready(
                container_name=self.container_name,
                db_user=self.db_user,
                db_name=self.db_name,
            )

    def stop(self) -> None:
        self.verify_exists()
        if self.status != ContainerStatus.STOPPED:
            call_docker_stop(self.container_name)
            self.status = ContainerStatus.STOPPED

    def destroy(self) -> None:
        self.verify_exists()
        call_docker_destroy(self.container_name, self.volume_name)

    def backup(self, output_dir: Path | None = None) -> Path:
        if not self.running:
            raise RuntimeError(
                f"Project '{self.name}' is {self.status} — start it before backing up"
            )

        backup_dir = (output_dir or self.default_backup_dir) / self.name
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{self.name}_{timestamp}.sql.gz"

        result = call_docker_pg_dump(
            container_name=self.container_name,
            db_user=self.db_user,
            db_name=self.db_name,
        )
        with gzip.open(backup_file, "wb") as f:
            f.write(result.stdout)

        return backup_file

    def restore(self, backup_file: Path) -> None:
        if not self.running:
            raise RuntimeError(
                f"Project '{self.name}' is {self.status} — start it before restoring"
            )

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        if backup_file.suffix == ".gz":
            with gzip.open(backup_file, "rb") as f:
                sql_bytes = f.read()
        else:
            sql_bytes = backup_file.read_bytes()

        docker_swap_in_db(
            sql_bytes=sql_bytes,
            container_name=self.container_name,
            db_user=self.db_user,
            target_db_name=self.db_name,
        )
