from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Self

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


class DockerProjectLabelFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    port: int | None
    created_at: datetime | None


class DockerProjectLabelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_prefix: ClassVar[str] = LABEL_PREFIX
    name_label_suffix: ClassVar[str] = ".name"
    port_label_suffix: ClassVar[str] = ".port"
    created_at_label_suffix: ClassVar[str] = ".created-at"
    name_label_key: ClassVar[str] = f"{label_prefix}{name_label_suffix}"
    port_label_key: ClassVar[str] = f"{label_prefix}{port_label_suffix}"
    created_at_label_key: ClassVar[str] = f"{label_prefix}{created_at_label_suffix}"

    name: str
    port: int | None = None
    created_at: datetime | None = None

    @classmethod
    def _label_fields(cls, labels: dict[str, str]) -> DockerProjectLabelFields:
        return DockerProjectLabelFields(
            name=labels[cls.name_label_key],
            port=cls._parse_port(labels.get(cls.port_label_key)),
            created_at=cls._parse_created_at(labels.get(cls.created_at_label_key)),
        )

    @classmethod
    def from_labels(cls, labels: dict[str, str]) -> Self:
        return cls(**cls._label_fields(labels).model_dump())

    def to_labels(self) -> dict[str, str]:
        labels = {self.name_label_key: self.name}
        if self.port is not None:
            labels[self.port_label_key] = str(self.port)
        if self.created_at is not None:
            labels[self.created_at_label_key] = self.created_at.isoformat()
        return labels

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


class DockerProjectMetadata(DockerProjectLabelMetadata):
    inspect_delimiter: ClassVar[str] = "||"

    status: ContainerStatus = ContainerStatus.UNKNOWN

    @classmethod
    def inspect_format(cls) -> str:
        return (
            "{{json .Config.Labels}}"
            f"{cls.inspect_delimiter}"
            "{{json .State.Status}}"
        )

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
            **cls._label_fields(labels).model_dump(),
            status=ContainerStatus.from_docker(status)
            if status
            else ContainerStatus.UNKNOWN,
        )


class DockerProjectCreateMetadata(DockerProjectLabelMetadata):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int
    created_at: datetime

    def docker_run_args(self) -> list[str]:
        label_args = [
            arg
            for key, value in self.to_labels().items()
            for arg in ("--label", f"{key}={value}")
        ]
        return [
            "-p",
            f"{self.port}:5432",
            *label_args,
        ]
