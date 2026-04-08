from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Generator
from contextlib import contextmanager, suppress
from time import sleep
from typing import IO, Literal, overload
from uuid import uuid4

from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
    DockerProjectMetadata,
)
from dr_llm.project.errors import (
    DockerCommandError,
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerContainerNotRunningError,
    DockerPortAllocatedError,
    DockerUnavailableError,
    ProjectError,
)

_VALID_PG_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_pg_identifier(name: str, label: str = "identifier") -> str:
    """Validate a string is safe for use as a PostgreSQL identifier."""
    if not _VALID_PG_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid PostgreSQL {label}: {name!r} "
            f"(must match {_VALID_PG_IDENTIFIER.pattern})"
        )
    return name


STREAM_CHUNK_SIZE = 1024 * 1024


def _docker_error_detail(stderr: str) -> str:
    detail = stderr.strip()
    return detail or "unknown docker error"


def _docker_error(args: tuple[str, ...], stderr: str) -> ProjectError:
    lowered = stderr.lower()
    if (
        "cannot connect to the docker daemon" in lowered
        or "error during connect" in lowered
    ):
        return DockerUnavailableError()
    if "no such container" in lowered:
        return DockerContainerNotFoundError()
    if "is not running" in lowered or "container is not running" in lowered:
        return DockerContainerNotRunningError()
    if "container name" in lowered and "is already in use" in lowered:
        return DockerContainerConflictError()
    if (
        "port is already allocated" in lowered
        or "bind: address already in use" in lowered
    ):
        return DockerPortAllocatedError()
    command = " ".join(["docker", *args])
    detail = _docker_error_detail(stderr)
    return DockerCommandError(f"Docker command failed: {command}\n{detail}")


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
            return ContainerStatus.RUNNING
        lowered_stderr = result.stderr.lower()
        if (
            "no such container" in lowered_stderr
            or "cannot connect to the docker daemon" in lowered_stderr
            or "error during connect" in lowered_stderr
        ):
            raise _docker_error(
                (
                    "exec",
                    container_name,
                    "pg_isready",
                    "-U",
                    db_user,
                    "-d",
                    db_name,
                ),
                result.stderr.strip(),
            )
        sleep(1)
    raise DockerCommandError(
        f"Postgres in {container_name} did not become ready within {timeout_seconds}s"
    )


@overload
def _call_docker_impl(
    *args: str,
    check: bool,
    text: Literal[True],
    input: None = None,
) -> subprocess.CompletedProcess[str]: ...


@overload
def _call_docker_impl(
    *args: str,
    check: bool,
    text: Literal[False],
    input: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]: ...


def _call_docker_impl(
    *args: str,
    check: bool,
    text: bool,
    input: bytes | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["docker", *args],
            input=input,
            capture_output=True,
            text=text,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError() from exc
    if check and result.returncode != 0:
        stderr = (
            result.stderr.strip()
            if text
            else result.stderr.decode(errors="replace").strip()
        )
        raise _docker_error(args, stderr)
    return result


def call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _call_docker_impl(*args, check=check, text=True)


def call_docker_bytes(
    *args: str,
    check: bool = True,
    input: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return _call_docker_impl(*args, check=check, text=False, input=input)


def _copy_binary_stream(
    source: IO[bytes],
    dest: IO[bytes],
) -> None:
    while chunk := source.read(STREAM_CHUNK_SIZE):
        dest.write(chunk)


def _read_process_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode(errors="replace").strip()


@contextmanager
def _temp_environ(**updates: str) -> Generator[None, None, None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


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
        error = _docker_error(
            (
                "inspect",
                "--format",
                "{{json .Config.Labels}}||{{json .State.Status}}",
                container_name,
            ),
            result.stderr.strip(),
        )
        if isinstance(error, DockerContainerNotFoundError):
            return None
        raise error
    return DockerProjectMetadata.from_inspect_output(
        result.stdout.strip(),
        label_prefix=label_prefix,
    )


def create_project_container(
    volume_name: str,
    container_name: str,
    db_name: str,
    db_user: str,
    db_password: str,
    docker_image: str,
    project: DockerProjectCreateMetadata,
) -> None:
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
    docker_cmd.extend(project.docker_run_args())
    docker_cmd.append(docker_image)
    with _temp_environ(POSTGRES_PASSWORD=db_password):
        call_docker(*docker_cmd)


def call_docker_start(container_name: str) -> None:
    result = call_docker("start", container_name, check=False)
    if result.returncode == 0 or "already running" in result.stderr.lower():
        return
    raise _docker_error(("start", container_name), result.stderr.strip())


def call_docker_stop(container_name: str) -> None:
    result = call_docker("stop", container_name, check=False)
    if result.returncode == 0 or "is not running" in result.stderr.lower():
        return
    raise _docker_error(("stop", container_name), result.stderr.strip())


def call_docker_destroy(container_name: str, volume_name: str) -> None:
    remove_container = call_docker("rm", "-f", container_name, check=False)
    remove_volume = call_docker("volume", "rm", volume_name, check=False)
    if remove_container.returncode != 0:
        raise _docker_error(
            ("rm", "-f", container_name), remove_container.stderr.strip()
        )
    if remove_volume.returncode != 0:
        raise _docker_error(("volume", "rm", volume_name), remove_volume.stderr.strip())


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


def _call_docker_psql_admin(
    container_name: str,
    db_user: str,
    *sql_commands: str,
) -> subprocess.CompletedProcess[bytes]:
    psql_args: list[str] = []
    for sql in sql_commands:
        psql_args.extend(["-c", sql])
    return call_docker_psql(
        container_name,
        db_user,
        "postgres",
        *psql_args,
    )


def call_docker_psql_input_stream(
    container_name: str,
    db_user: str,
    db_name: str,
    sql_stream: IO[bytes],
) -> None:
    args = (
        "exec",
        "-i",
        container_name,
        "psql",
        "-U",
        db_user,
        db_name,
    )
    try:
        process = subprocess.Popen(
            ["docker", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError() from exc
    assert process.stdin is not None
    try:
        _copy_binary_stream(sql_stream, process.stdin)
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        with suppress(BrokenPipeError):
            process.stdin.close()

    stderr = _read_process_stderr(process)
    if process.wait() != 0:
        raise _docker_error(args, stderr)


def docker_swap_in_db(
    sql_stream: IO[bytes],
    container_name: str,
    db_user: str,
    target_db_name: str,
) -> None:
    swap_in_db = f"dr_llm_restore_{uuid4().hex[:8]}"
    _validate_pg_identifier(swap_in_db, "database name")
    _validate_pg_identifier(target_db_name, "database name")
    _call_docker_psql_admin(
        container_name,
        db_user,
        f'CREATE DATABASE "{swap_in_db}";',
    )
    try:
        call_docker_psql_input_stream(container_name, db_user, swap_in_db, sql_stream)
        _call_docker_psql_admin(
            container_name,
            db_user,
            f'DROP DATABASE IF EXISTS "{target_db_name}";',
            f'ALTER DATABASE "{swap_in_db}" RENAME TO "{target_db_name}";',
        )
    except Exception:
        _call_docker_psql_admin(
            container_name,
            db_user,
            f'DROP DATABASE IF EXISTS "{swap_in_db}";',
        )
        raise


def call_docker_pg_dump_stream(
    container_name: str,
    db_user: str,
    db_name: str,
    output_stream: IO[bytes],
) -> None:
    args = (
        "exec",
        container_name,
        "pg_dump",
        "-U",
        db_user,
        db_name,
    )
    try:
        process = subprocess.Popen(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError() from exc
    assert process.stdout is not None
    try:
        _copy_binary_stream(process.stdout, output_stream)
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()

    stderr = _read_process_stderr(process)
    if process.wait() != 0:
        raise _docker_error(args, stderr)


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
    if result.returncode != 0:
        raise _docker_error(
            (
                "ps",
                "-a",
                "--filter",
                f"label={label_prefix}.name",
                "--format",
                "{{json .}}",
            ),
            result.stderr.strip(),
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
