from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

LABEL_PREFIX = "dr-llm.project"


class ContainerStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

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

    label_prefix: ClassVar[str] = LABEL_PREFIX
    inspect_delimiter: ClassVar[str] = "||"
    name_label_suffix: ClassVar[str] = ".name"
    port_label_suffix: ClassVar[str] = ".port"
    created_at_label_suffix: ClassVar[str] = ".created-at"
    name_label_key: ClassVar[str] = f"{label_prefix}{name_label_suffix}"
    port_label_key: ClassVar[str] = f"{label_prefix}{port_label_suffix}"
    created_at_label_key: ClassVar[str] = f"{label_prefix}{created_at_label_suffix}"

    name: str
    port: int | None = None
    created_at: datetime | None = None
    status: ContainerStatus = ContainerStatus.UNKNOWN

    @classmethod
    def from_inspect_output(
        cls,
        raw: str,
    ) -> DockerProjectMetadata:
        labels_raw, status_raw = raw.split(cls.inspect_delimiter, 1)
        return cls.from_labels_status(
            labels=json.loads(labels_raw),
            status=json.loads(status_raw) if status_raw else None,
        )

    @classmethod
    def from_labels_status(
        cls,
        *,
        labels: dict[str, str],
        status: str | None,
    ) -> DockerProjectMetadata:
        return cls(
            name=labels[cls.name_label_key],
            port=cls._parse_port(labels.get(cls.port_label_key)),
            created_at=cls._parse_created_at(labels.get(cls.created_at_label_key)),
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

    name: str
    port: int
    created_at: datetime

    def docker_run_args(self) -> list[str]:
        return [
            "-p",
            f"{self.port}:5432",
            "--label",
            f"{DockerProjectMetadata.name_label_key}={self.name}",
            "--label",
            f"{DockerProjectMetadata.port_label_key}={self.port}",
            "--label",
            f"{DockerProjectMetadata.created_at_label_key}={self.created_at.isoformat()}",
        ]
