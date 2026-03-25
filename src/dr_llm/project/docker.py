from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from time import sleep

from dr_llm.project.models import (
    DB_NAME,
    DB_PASSWORD,
    DB_USER,
    DOCKER_IMAGE,
    LABEL_PREFIX,
    ProjectInfo,
    container_name,
    dsn_for_port,
    parse_docker_labels,
    volume_name,
)
from dr_llm.project.ports import find_available_port


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _require_docker() -> None:
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Docker is not available. Install Docker or start the daemon."
        )


def _wait_ready(cname: str, timeout_seconds: int = 30) -> None:
    for _ in range(timeout_seconds):
        result = subprocess.run(
            ["docker", "exec", cname, "pg_isready", "-U", DB_USER, "-d", DB_NAME],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            sleep(1)
            return
        sleep(1)
    raise RuntimeError(
        f"Postgres in {cname} did not become ready within {timeout_seconds}s"
    )


def _apply_schema(port: int) -> None:
    from dr_llm.storage import PostgresRepository, StorageConfig

    dsn = dsn_for_port(port)
    repo = PostgresRepository(StorageConfig(dsn=dsn))
    try:
        repo.initialize()
    finally:
        repo.close()


def _inspect_container(cname: str) -> dict[str, str] | None:
    result = _docker(
        "inspect",
        "--format",
        "{{json .Config.Labels}}||{{json .State.Status}}",
        cname,
        check=False,
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split("||", 1)
    labels = json.loads(parts[0]) if parts[0] else {}
    status = json.loads(parts[1]) if len(parts) > 1 else "unknown"
    labels["__status__"] = status
    return labels


def _info_from_labels(name: str, labels: dict[str, str]) -> ProjectInfo:
    port = int(labels.get(f"{LABEL_PREFIX}.port", "0"))
    status = labels.get("__status__", "unknown")
    return ProjectInfo(
        name=name,
        container_name=container_name(name),
        volume_name=volume_name(name),
        port=port,
        status=status,
        dsn=dsn_for_port(port),
        created_at=labels.get(f"{LABEL_PREFIX}.created-at"),
    )


def create_project(name: str) -> ProjectInfo:
    _require_docker()
    cname = container_name(name)
    vname = volume_name(name)

    existing = _inspect_container(cname)
    if existing is not None:
        raise RuntimeError(f"Project '{name}' already exists (container {cname})")

    port = find_available_port()
    now = datetime.now(timezone.utc).isoformat()

    _docker("volume", "create", vname)
    _docker(
        "run",
        "-d",
        "--name",
        cname,
        "-v",
        f"{vname}:/var/lib/postgresql/data",
        "-e",
        f"POSTGRES_DB={DB_NAME}",
        "-e",
        f"POSTGRES_USER={DB_USER}",
        "-e",
        f"POSTGRES_PASSWORD={DB_PASSWORD}",
        "-p",
        f"{port}:5432",
        "--label",
        f"{LABEL_PREFIX}.name={name}",
        "--label",
        f"{LABEL_PREFIX}.port={port}",
        "--label",
        f"{LABEL_PREFIX}.created-at={now}",
        DOCKER_IMAGE,
    )

    _wait_ready(cname)
    _apply_schema(port)

    return ProjectInfo(
        name=name,
        container_name=cname,
        volume_name=vname,
        port=port,
        status="running",
        dsn=dsn_for_port(port),
        created_at=now,
    )


def start_project(name: str) -> ProjectInfo:
    _require_docker()
    cname = container_name(name)
    labels = _inspect_container(cname)
    if labels is None:
        raise RuntimeError(f"Project '{name}' not found")
    if labels["__status__"] == "running":
        return _info_from_labels(name, labels)

    _docker("start", cname)
    _wait_ready(cname)
    labels["__status__"] = "running"
    return _info_from_labels(name, labels)


def stop_project(name: str) -> None:
    _require_docker()
    cname = container_name(name)
    labels = _inspect_container(cname)
    if labels is None:
        raise RuntimeError(f"Project '{name}' not found")
    _docker("stop", cname)


def destroy_project(name: str) -> None:
    _require_docker()
    cname = container_name(name)
    vname = volume_name(name)
    _docker("rm", "-f", cname, check=False)
    _docker("volume", "rm", vname, check=False)


def list_projects() -> list[ProjectInfo]:
    _require_docker()
    result = _docker(
        "ps",
        "-a",
        "--filter",
        f"label={LABEL_PREFIX}.name",
        "--format",
        "{{json .}}",
        check=False,
    )
    projects: list[ProjectInfo] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        data = json.loads(line)
        labels = parse_docker_labels(data.get("Labels", ""))
        pname = labels.get(f"{LABEL_PREFIX}.name")
        if pname is None:
            continue
        status = data.get("State", "unknown")
        labels["__status__"] = status
        projects.append(_info_from_labels(pname, labels))
    return projects


def get_project(name: str) -> ProjectInfo | None:
    _require_docker()
    cname = container_name(name)
    labels = _inspect_container(cname)
    if labels is None:
        return None
    return _info_from_labels(name, labels)
