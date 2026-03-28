from __future__ import annotations

import json
import subprocess
from enum import StrEnum
from time import sleep
from typing import Any
from uuid import uuid4


class ContainerStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @classmethod
    def default(cls) -> ContainerStatus:
        return cls.UNKNOWN


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
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f"CREATE DATABASE {db_name};",
    )


def call_docker_psql_drop_db(
    container_name: str,
    db_user: str,
    db_name: str,
) -> subprocess.CompletedProcess[bytes]:
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f"DROP DATABASE IF EXISTS {db_name};",
    )


def call_docker_psql_swap_in_db(
    container_name: str,
    db_user: str,
    target_db_name: str,
    swap_in_db: str,
) -> subprocess.CompletedProcess[bytes]:
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        "-c",
        f"DROP DATABASE IF EXISTS {target_db_name};",
        "-c",
        f"ALTER DATABASE {swap_in_db} RENAME TO {target_db_name};",
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
    except subprocess.CalledProcessError:
        call_docker_psql_drop_db(container_name, db_user, swap_in_db)
        raise
    call_docker_psql_swap_in_db(
        container_name,
        db_user,
        target_db_name,
        swap_in_db,
    )


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
