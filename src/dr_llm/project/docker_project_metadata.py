from __future__ import annotations

import json
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
    created_at: str | None = None
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
        labels, status = cls._parse_inspect_output(raw)
        return cls.from_labels_status(
            labels=labels,
            status=status,
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
            created_at=labels.get(cls.created_at_key(label_prefix)),
            status=ContainerStatus.from_docker(status)
            if status
            else ContainerStatus.default(),
        )

    @classmethod
    def _parse_inspect_output(cls, raw: str) -> tuple[dict[str, str], str | None]:
        labels_raw, status_raw = raw.split(cls.inspect_delimiter, 1)
        labels = json.loads(labels_raw)
        status = json.loads(status_raw) if status_raw else None
        return labels, status

    @staticmethod
    def _parse_port(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        return int(value)
