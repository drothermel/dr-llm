from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, computed_field

from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.docker_psql import _validate_pg_identifier


class ProjectInfo(BaseModel):
    db_name: ClassVar[str] = _validate_pg_identifier("dr_llm", "database name")
    db_user: ClassVar[str] = "postgres"
    db_password: ClassVar[str] = "postgres"
    container_prefix: ClassVar[str] = "dr-llm-pg-"
    volume_prefix: ClassVar[str] = "dr-llm-data-"

    name: str
    port: int | None = None
    status: ContainerStatus = ContainerStatus.UNKNOWN
    created_at: datetime | None = None

    @computed_field
    @property
    def dsn(self) -> str | None:
        if self.port is None:
            return None
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@localhost:{self.port}/{self.db_name}"
        )

    @computed_field
    @property
    def volume_name(self) -> str:
        return f"{self.volume_prefix}{self.name}"

    @computed_field
    @property
    def container_name(self) -> str:
        return f"{self.container_prefix}{self.name}"

    @computed_field
    @property
    def running(self) -> bool:
        return self.status == ContainerStatus.RUNNING
