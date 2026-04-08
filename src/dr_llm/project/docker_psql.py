from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import IO, cast
from uuid import uuid4

from dr_llm.project.docker_runner import (
    BinaryStream,
    call_docker_bytes,
    copy_binary_stream,
    docker_error,
    read_process_stderr,
)
from dr_llm.project.errors import DockerUnavailableError

_VALID_PG_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_pg_identifier(name: str, label: str = "identifier") -> str:
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


def _run_docker_process(
    args: tuple[str, ...],
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    operation: Callable[[subprocess.Popen[bytes]], None],
) -> None:
    try:
        process = subprocess.Popen(
            ["docker", *args],
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError() from exc
    stderr_chunks: list[str] = []

    def read_stderr() -> None:
        stderr_chunks.append(read_process_stderr(process))

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    try:
        operation(process)
    except Exception:
        process.kill()
        process.wait()
        stderr_thread.join()
        raise

    stderr_thread.join()
    stderr = "".join(stderr_chunks)
    if process.wait() != 0:
        raise docker_error(args, stderr)


def call_docker_psql_input_stream(
    container_name: str,
    db_user: str,
    db_name: str,
    sql_stream: BinaryStream,
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

    def write_sql(process: subprocess.Popen[bytes]) -> None:
        process_stdin = cast(IO[bytes], process.stdin)
        try:
            with suppress(BrokenPipeError):
                copy_binary_stream(sql_stream, process_stdin)
        finally:
            with suppress(BrokenPipeError):
                process_stdin.close()

    _run_docker_process(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        operation=write_sql,
    )


def call_docker_pg_dump_stream(
    container_name: str,
    db_user: str,
    db_name: str,
    output_stream: BinaryStream,
) -> None:
    args = (
        "exec",
        container_name,
        "pg_dump",
        "-U",
        db_user,
        db_name,
    )

    def read_dump(process: subprocess.Popen[bytes]) -> None:
        process_stdout = cast(IO[bytes], process.stdout)
        try:
            copy_binary_stream(process_stdout, output_stream)
        finally:
            process_stdout.close()

    _run_docker_process(
        args,
        stdout=subprocess.PIPE,
        operation=read_dump,
    )


def docker_swap_in_db(
    sql_stream: BinaryStream,
    container_name: str,
    db_user: str,
    target_db_name: str,
) -> None:
    swap_in_db = f"dr_llm_restore_{uuid4().hex[:8]}"
    validate_pg_identifier(swap_in_db, "database name")
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
    except Exception as exc:
        _drop_swap_db_noting_cleanup_failure(
            container_name,
            db_user,
            swap_in_db,
            original_exc=exc,
        )
        raise


def _drop_swap_db_noting_cleanup_failure(
    container_name: str,
    db_user: str,
    swap_in_db: str,
    *,
    original_exc: BaseException,
) -> None:
    try:
        _call_docker_psql_admin(
            container_name,
            db_user,
            f'DROP DATABASE IF EXISTS "{swap_in_db}";',
        )
    except Exception as cleanup_exc:
        original_exc.add_note(
            f"Cleanup of temporary restore database {swap_in_db!r} also failed: "
            f"{cleanup_exc}"
        )
