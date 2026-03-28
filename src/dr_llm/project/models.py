from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, computed_field

from dr_llm.project.docker import (
    ContainerStatus,
    call_docker_create,
    call_docker_destroy,
    call_docker_get_labels_status,
    call_docker_start,
    call_docker_stop,
    wait_docker_ready,
)
from dr_llm.project.ports import find_available_port
from dr_llm.storage.repository import try_init_repo_from_dsn


class ProjectInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    db_name: ClassVar[str] = "dr_llm"
    db_user: ClassVar[str] = "postgres"
    db_password: ClassVar[str] = "postgres"
    docker_image: ClassVar[str] = "postgres:16"
    container_prefix: ClassVar[str] = "dr-llm-pg-"
    volume_prefix: ClassVar[str] = "dr-llm-data-"
    label_prefix: ClassVar[str] = "dr-llm.project"

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
        return self.get_containter_name(self.name)

    @computed_field
    @property
    def running(self) -> bool:
        return self.status == ContainerStatus.RUNNING

    @classmethod
    def port_key(cls) -> str:
        return f"{cls.label_prefix}.port"

    @classmethod
    def created_at_key(cls) -> str:
        return f"{cls.label_prefix}.created-at"

    @classmethod
    def get_volume_name(cls, name: str) -> str:
        return f"{cls.volume_prefix}{name}"

    @classmethod
    def get_containter_name(cls, name: str) -> str:
        return f"{cls.container_prefix}{name}"

    @classmethod
    def get_dsn(cls, port: int) -> str:
        return f"postgresql://{cls.db_user}:{cls.db_password}@localhost:{port}/{cls.db_name}"

    @classmethod
    def check_exists(cls, container_name: str) -> bool:
        return call_docker_get_labels_status(container_name) is not None

    @classmethod
    def from_labels(
        cls, name: str, labels: dict[str, str], status: str | None
    ) -> ProjectInfo:
        return ProjectInfo(
            name=name,
            port=int(labels.get(cls.port_key(), "0")),
            status=ContainerStatus(status or ContainerStatus.default()),
            created_at=labels.get(cls.created_at_key()),
        )

    @classmethod
    def maybe_from_existing(cls, name: str) -> ProjectInfo | None:
        container_name = cls.get_containter_name(name)
        raw_labels_status = call_docker_get_labels_status(container_name)
        if raw_labels_status is None:
            return None

        res_split = raw_labels_status.split(" || ", 1)
        if len(res_split) == 0 or res_split[0] is None:
            raise RuntimeError(f"Labels for project '{name}' incorrect shape")

        status = None
        labels = json.loads(res_split[0])
        if len(res_split) > 1 and res_split[1] is not None:
            status = json.loads(res_split[1])
        return cls.from_labels(name, labels, status)

    @classmethod
    def get_by_name(cls, name: str) -> ProjectInfo:
        project_info = cls.maybe_from_existing(name)
        if project_info is None:
            raise RuntimeError(f"Project '{name}' not found")
        return project_info

    @classmethod
    def create_new(cls, name: str) -> ProjectInfo:
        project_info = cls(
            name=name,
            port=find_available_port(),
            created_at=datetime.now(timezone.utc).isoformat(),
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
        try_init_repo_from_dsn(project_info.dsn)
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
