from __future__ import annotations

import json
import os
import re
import subprocess
from enum import StrEnum
from time import sleep
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

_VALID_PG_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_pg_identifier(name: str, label: str = "identifier") -> str:
    """Validate a string is safe for use as a PostgreSQL identifier."""
    if not _VALID_PG_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid PostgreSQL {label}: {name!r} "
            f"(must match {_VALID_PG_IDENTIFIER.pattern})"
        )
    return name


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


def _docker_error(args: tuple[str, ...], stderr: str) -> RuntimeError:
    lowered = stderr.lower()
    if (
        "cannot connect to the docker daemon" in lowered
        or "error during connect" in lowered
    ):
        return RuntimeError(
            "Docker is not available. Install Docker or start the daemon."
        )
    command = " ".join(["docker", *args])
    detail = stderr or "unknown docker error"
    return RuntimeError(f"Docker command failed: {command}\n{detail}")


def wait_docker_ready(
    container_name: str,
    db_user: str,
    db_name: str,
    timeout_seconds: int = 30,
) -> ContainerStatus:
    for _ in range(timeout_seconds):
        result = call_docker(
            "exec",
            container_name,
            "pg_isready",
            "-U",
            db_user,
            "-d",
            db_name,
            check=False,
        )
        if result.returncode == 0:
            sleep(1)
            return ContainerStatus.RUNNING
        sleep(1)
    raise RuntimeError(
        f"Postgres in {container_name} did not become ready within {timeout_seconds}s"
    )


def call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker is not available. Install Docker or start the daemon."
        ) from exc
    if check and result.returncode != 0:
        raise _docker_error(args, result.stderr.strip())
    return result


def call_docker_bytes(
    *args: str,
    check: bool = True,
    input: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["docker", *args],
            input=input,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker is not available. Install Docker or start the daemon."
        ) from exc
    if check and result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise _docker_error(args, stderr)
    return result


def get_docker_project_metadata(
    container_name: str,
    *,
    label_prefix: str,
) -> DockerProjectMetadata | None:
    result = call_docker(
        "inspect",
        "--format",
        "{{json .Config.Labels}}||{{json .State.Status}}",
        container_name,
        check=False,
    )
    if result.returncode != 0:
        return None
    return DockerProjectMetadata.from_inspect_output(
        result.stdout.strip(),
        label_prefix=label_prefix,
    )


def call_docker_create(
    volume_name: str,
    container_name: str,
    db_name: str,
    db_user: str,
    db_password: str,
    docker_image: str,
    label_prefix: str | None = None,
    name: str | None = None,
    port: int | None = None,
    created_at: str | None = None,
):
    call_docker("volume", "create", volume_name)
    docker_cmd = [
        "run",
        "-d",
        "--name",
        container_name,
        "-v",
        f"{volume_name}:/var/lib/postgresql/data",
        "-e",
        f"POSTGRES_DB={db_name}",
        "-e",
        f"POSTGRES_USER={db_user}",
        "-e",
        "POSTGRES_PASSWORD",
    ]
    if port is not None:
        docker_cmd.extend(["-p", f"{port}:5432"])
    if all(val is not None for val in [label_prefix, name, port, created_at]):
        docker_cmd.extend(
            [
                "--label",
                f"{label_prefix}.name={name}",
                "--label",
                f"{label_prefix}.port={port}",
                "--label",
                f"{label_prefix}.created-at={created_at}",
            ]
        )
    docker_cmd.append(docker_image)
    prev = os.environ.get("POSTGRES_PASSWORD")
    os.environ["POSTGRES_PASSWORD"] = db_password
    try:
        call_docker(*docker_cmd)
    finally:
        if prev is None:
            os.environ.pop("POSTGRES_PASSWORD", None)
        else:
            os.environ["POSTGRES_PASSWORD"] = prev


def call_docker_start(container_name: str) -> None:
    call_docker("start", container_name)


def call_docker_stop(container_name: str) -> None:
    call_docker("stop", container_name)


def call_docker_destroy(container_name: str, volume_name: str) -> None:
    call_docker("rm", "-f", container_name, check=False)
    call_docker("volume", "rm", volume_name, check=False)


def call_docker_psql(
    container_name: str,
    db_user: str,
    db_name: str,
    *args: str,
) -> subprocess.CompletedProcess[bytes]:
    return call_docker_bytes(
        "exec",
        container_name,
        "psql",
        "-U",
        db_user,
        db_name,
        *args,
    )


def call_docker_psql_input(
    container_name: str,
    db_user: str,
    db_name: str,
    sql_bytes: bytes,
) -> subprocess.CompletedProcess[bytes]:
    return call_docker_bytes(
        "exec",
        "-i",
        container_name,
        "psql",
        "-U",
        db_user,
        db_name,
        input=sql_bytes,
    )


def call_docker_psql_create_db(
    container_name: str,
    db_user: str,
    db_name: str,
) -> subprocess.CompletedProcess[bytes]:
    _validate_pg_identifier(db_name, "database name")
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f'CREATE DATABASE "{db_name}";',
    )


def call_docker_psql_drop_db(
    container_name: str,
    db_user: str,
    db_name: str,
) -> subprocess.CompletedProcess[bytes]:
    _validate_pg_identifier(db_name, "database name")
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f'DROP DATABASE IF EXISTS "{db_name}";',
    )


def call_docker_psql_swap_in_db(
    container_name: str,
    db_user: str,
    target_db_name: str,
    swap_in_db: str,
) -> subprocess.CompletedProcess[bytes]:
    _validate_pg_identifier(target_db_name, "database name")
    _validate_pg_identifier(swap_in_db, "database name")
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f'DROP DATABASE IF EXISTS "{target_db_name}";',
        "-c",
        f'ALTER DATABASE "{swap_in_db}" RENAME TO "{target_db_name}";',
    )


def docker_swap_in_db(
    sql_bytes: bytes,
    container_name: str,
    db_user: str,
    target_db_name: str,
) -> None:
    swap_in_db = f"dr_llm_restore_{uuid4().hex[:8]}"
    call_docker_psql_create_db(container_name, db_user, swap_in_db)
    try:
        call_docker_psql_input(container_name, db_user, swap_in_db, sql_bytes)
        call_docker_psql_swap_in_db(
            container_name,
            db_user,
            target_db_name,
            swap_in_db,
        )
    except Exception:
        call_docker_psql_drop_db(container_name, db_user, swap_in_db)
        raise


def call_docker_pg_dump(
    container_name: str,
    db_user: str,
    db_name: str,
) -> subprocess.CompletedProcess[bytes]:
    return call_docker_bytes(
        "exec",
        container_name,
        "pg_dump",
        "-U",
        db_user,
        db_name,
    )


def parse_docker_labels(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    if raw.startswith('"'):
        raw = json.loads(raw)
    labels: dict[str, str] = {}
    for pair in raw.split(","):
        k, _, v = pair.partition("=")
        labels[k.strip()] = v.strip()
    return labels


def call_docker_list_labels(label_prefix: str) -> str:
    result = call_docker(
        "ps",
        "-a",
        "--filter",
        f"label={label_prefix}.name",
        "--format",
        "{{json .}}",
        check=False,
    )
    return result.stdout.strip()


def get_claimed_project_ports(label_prefix: str) -> set[int]:
    ports: set[int] = set()
    for metadata in get_all_docker_project_metadata(label_prefix):
        if metadata.port is not None:
            ports.add(metadata.port)
    return ports


def get_all_docker_project_metadata(label_prefix: str) -> list[DockerProjectMetadata]:
    project_metadata: list[DockerProjectMetadata] = []
    for line in call_docker_list_labels(label_prefix).splitlines():
        if not line:
            continue
        data = json.loads(line)
        status = data.get("State")
        parsed = parse_docker_labels(data.get("Labels", ""))
        if DockerProjectMetadata.name_key(label_prefix) not in parsed:
            continue
        project_metadata.append(
            DockerProjectMetadata.from_labels_status(
                labels=parsed,
                status=status,
                label_prefix=label_prefix,
            )
        )
    return project_metadata
