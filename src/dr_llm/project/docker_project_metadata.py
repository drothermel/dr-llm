from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Self

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
    def _label_kwargs(cls, labels: dict[str, str]) -> dict[str, Any]:
        return {
            "name": labels[cls.name_label_key],
            "port": labels.get(cls.port_label_key),
            "created_at": labels.get(cls.created_at_label_key),
        }

    @classmethod
    def from_labels(cls, labels: dict[str, str]) -> Self:
        return cls(**cls._label_kwargs(labels))

    def to_labels(self) -> dict[str, str]:
        labels = {self.name_label_key: self.name}
        if self.port is not None:
            labels[self.port_label_key] = str(self.port)
        if self.created_at is not None:
            labels[self.created_at_label_key] = self.created_at.isoformat()
        return labels


class DockerProjectMetadata(DockerProjectLabelMetadata):
    status: ContainerStatus = ContainerStatus.UNKNOWN

    @classmethod
    def inspect_format(cls) -> str:
        return "[{{json .Config.Labels}},{{json .State.Status}}]"

    @classmethod
    def from_inspect_output(cls, raw: str) -> DockerProjectMetadata:
        labels, status = json.loads(raw)
        return cls.from_labels_status(labels=labels, status=status)

    @classmethod
    def from_labels_status(
        cls,
        *,
        labels: dict[str, str],
        status: str | None,
    ) -> DockerProjectMetadata:
        return cls(
            **cls._label_kwargs(labels),
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
