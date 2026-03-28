from __future__ import annotations

import json
import subprocess
from enum import StrEnum
from time import sleep
from typing import Any


class ContainerStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @classmethod
    def default(cls) -> ContainerStatus:
        return cls.UNKNOWN


def require_docker() -> None:
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Docker is not available. Install Docker or start the daemon."
        )


def wait_docker_ready(
    container_name: str,
    db_user: str,
    db_name: str,
    timeout_seconds: int = 30,
) -> ContainerStatus:
    for _ in range(timeout_seconds):
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "pg_isready",
                "-U",
                db_user,
                "-d",
                db_name,
            ],
            capture_output=True,
            text=True,
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
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def call_docker_get_labels_status(container_name: str) -> str | None:
    result = call_docker(
        "inspect",
        "--format",
        "{{json .Config.Labels}}||{{json .State.Status}}",
        container_name,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


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
        f"POSTGRES_PASSWORD={db_password}",
        "-p",
        f"{port}:5432",
    ]
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
    call_docker(*docker_cmd)


def call_docker_start(container_name: str) -> None:
    call_docker("start", container_name)


def call_docker_stop(container_name: str) -> None:
    call_docker("stop", container_name)


def call_docker_destroy(container_name: str, volume_name: str) -> None:
    call_docker("rm", "-f", container_name, check=False)
    call_docker("volume", "rm", volume_name, check=False)


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


def get_all_docker_names_labels_status(label_prefix: str) -> list[dict[str, Any]]:
    names_labels_status = []
    for line in call_docker_list_labels(label_prefix).splitlines():
        if not line:
            continue
        data = json.loads(line)
        status = data.get("State")
        parsed = parse_docker_labels(data.get("Labels", ""))
        pname = parsed.get(f"{label_prefix}.name")
        if pname is None:
            continue
        names_labels_status.append(
            {
                "name": pname,
                "labels": parsed,
                "status": status,
            }
        )
    return names_labels_status
