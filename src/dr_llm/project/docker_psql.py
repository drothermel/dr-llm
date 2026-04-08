from __future__ import annotations

import re
import subprocess
from contextlib import suppress
from typing import IO
from uuid import uuid4

from dr_llm.project.docker_runner import (
    _copy_binary_stream,
    _docker_error,
    _read_process_stderr,
    call_docker_bytes,
)
from dr_llm.project.errors import DockerUnavailableError

_VALID_PG_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_pg_identifier(name: str, label: str = "identifier") -> str:
    if not _VALID_PG_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid PostgreSQL {label}: {name!r} "
            f"(must match {_VALID_PG_IDENTIFIER.pattern})"
        )
    return name


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


def docker_swap_in_db(
    sql_stream: IO[bytes],
    container_name: str,
    db_user: str,
    target_db_name: str,
) -> None:
    swap_in_db = f"dr_llm_restore_{uuid4().hex[:8]}"
    _validate_pg_identifier(swap_in_db, "database name")
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
