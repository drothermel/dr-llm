from __future__ import annotations

import os
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from typing import IO, Literal, overload

from dr_llm.project.errors import (
    DockerCommandError,
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerContainerNotRunningError,
    DockerPortAllocatedError,
    DockerUnavailableError,
    ProjectError,
)

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


def _run_or_error(
    *args: str,
    check_return: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = call_docker(*args, check=False)
    if check_return and result.returncode != 0:
        raise _docker_error(args, result.stderr.strip())
    return result


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
