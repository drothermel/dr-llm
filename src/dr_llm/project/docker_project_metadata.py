from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContainerStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @classmethod
    def default(cls) -> ContainerStatus:
        return cls.UNKNOWN

    @classmethod
    def from_docker(cls, status: str) -> ContainerStatus:
        """Map a Docker container status string to ContainerStatus."""
        if status == "running":
            return cls.RUNNING
        if status in ("exited", "created", "paused", "restarting", "removing", "dead"):
            return cls.STOPPED
        return cls.UNKNOWN


class DockerProjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspect_delimiter: ClassVar[str] = "||"

    name: str
    port: int | None = None
    created_at: datetime | None = None
    status: ContainerStatus = ContainerStatus.UNKNOWN

    @classmethod
    def name_key(cls, label_prefix: str) -> str:
        return f"{label_prefix}.name"

    @classmethod
    def port_key(cls, label_prefix: str) -> str:
        return f"{label_prefix}.port"

    @classmethod
    def created_at_key(cls, label_prefix: str) -> str:
        return f"{label_prefix}.created-at"

    @classmethod
    def from_inspect_output(
        cls,
        raw: str,
        *,
        label_prefix: str,
    ) -> DockerProjectMetadata:
        labels_raw, status_raw = raw.split(cls.inspect_delimiter, 1)
        return cls.from_labels_status(
            labels=json.loads(labels_raw),
            status=json.loads(status_raw) if status_raw else None,
            label_prefix=label_prefix,
        )

    @classmethod
    def from_labels_status(
        cls,
        *,
        labels: dict[str, str],
        status: str | None,
        label_prefix: str,
    ) -> DockerProjectMetadata:
        return cls(
            name=labels[cls.name_key(label_prefix)],
            port=cls._parse_port(labels.get(cls.port_key(label_prefix))),
            created_at=cls._parse_created_at(
                labels.get(cls.created_at_key(label_prefix))
            ),
            status=ContainerStatus.from_docker(status)
            if status
            else ContainerStatus.UNKNOWN,
        )

    @staticmethod
    def _parse_port(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _parse_created_at(value: str | None) -> datetime | None:
        if value is None or value == "":
            return None
        return datetime.fromisoformat(value)


class DockerProjectCreateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label_prefix: str
    name: str
    port: int
    created_at: datetime

    def docker_run_args(self) -> list[str]:
        return [
            "-p",
            f"{self.port}:5432",
            "--label",
            f"{DockerProjectMetadata.name_key(self.label_prefix)}={self.name}",
            "--label",
            f"{DockerProjectMetadata.port_key(self.label_prefix)}={self.port}",
            "--label",
            f"{DockerProjectMetadata.created_at_key(self.label_prefix)}={self.created_at.isoformat()}",
        ]
