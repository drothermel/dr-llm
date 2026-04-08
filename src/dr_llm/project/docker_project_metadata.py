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
    name_label_suffix: ClassVar[str] = ".name"
    port_label_suffix: ClassVar[str] = ".port"
    created_at_label_suffix: ClassVar[str] = ".created-at"

    name: str
    port: int | None = None
    created_at: datetime | None = None
    status: ContainerStatus = ContainerStatus.UNKNOWN

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
        name_key = f"{label_prefix}{cls.name_label_suffix}"
        port_key = f"{label_prefix}{cls.port_label_suffix}"
        created_at_key = f"{label_prefix}{cls.created_at_label_suffix}"
        return cls(
            name=labels[name_key],
            port=cls._parse_port(labels.get(port_key)),
            created_at=cls._parse_created_at(labels.get(created_at_key)),
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
        name_key = f"{self.label_prefix}{DockerProjectMetadata.name_label_suffix}"
        port_key = f"{self.label_prefix}{DockerProjectMetadata.port_label_suffix}"
        created_at_key = (
            f"{self.label_prefix}{DockerProjectMetadata.created_at_label_suffix}"
        )
        return [
            "-p",
            f"{self.port}:5432",
            "--label",
            f"{name_key}={self.name}",
            "--label",
            f"{port_key}={self.port}",
            "--label",
            f"{created_at_key}={self.created_at.isoformat()}",
        ]
