from __future__ import annotations

import os
from time import monotonic, sleep

import psycopg

from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
)
from dr_llm.project.docker_runner import (
    call_docker,
    docker_error,
    run_or_error,
)
from dr_llm.project.errors import (
    DockerCommandError,
    DockerContainerNotFoundError,
    DockerUnavailableError,
)


def wait_docker_ready(
    container_name: str,
    db_user: str,
    db_name: str,
    timeout_seconds: int = 30,
) -> ContainerStatus:
    args = (
        "exec",
        container_name,
        "pg_isready",
        "-U",
        db_user,
        "-d",
        db_name,
    )
    for _ in range(timeout_seconds):
        result = run_or_error(*args, check_return=False)
        if result.returncode == 0:
            return ContainerStatus.RUNNING
        error = docker_error(args, result.stderr.strip())
        if isinstance(error, (DockerContainerNotFoundError, DockerUnavailableError)):
            raise error
        sleep(1)
    raise DockerCommandError(
        f"Postgres in {container_name} did not become ready within {timeout_seconds}s"
    )


def wait_dsn_ready(
    dsn: str,
    *,
    timeout_seconds: int = 30,
    poll_interval_s: float = 0.5,
    connect_timeout_s: int = 2,
) -> None:
    """Probe the host-side Postgres TCP listener until ``SELECT 1`` succeeds.

    Complements :func:`wait_docker_ready`. ``pg_isready`` runs *inside* the
    container against the local Unix socket and reports ready as soon as
    the postmaster is up — but the host-mapped TCP listener can still be
    bouncing connections during init (e.g. while the postmaster finishes
    user/role setup or recovery). Callers that hand the published DSN to
    SQLAlchemy/psycopg right after :func:`wait_docker_ready` returns can
    hit a "server closed the connection unexpectedly" error.

    This helper opens a *fresh* psycopg connection to ``dsn`` and runs
    ``SELECT 1``, retrying every ``poll_interval_s`` until either the probe
    succeeds or ``timeout_seconds`` elapses. Use it after
    :func:`wait_docker_ready` so the container is known alive before we
    keep banging on the TCP listener.
    """
    deadline = monotonic() + timeout_seconds
    last_error: psycopg.OperationalError | None = None
    while True:
        try:
            with (
                psycopg.connect(dsn, connect_timeout=connect_timeout_s) as conn,
                conn.cursor() as cur,
            ):
                cur.execute("SELECT 1")
                cur.fetchone()
            return
        except psycopg.OperationalError as exc:
            last_error = exc
        if monotonic() >= deadline:
            raise DockerCommandError(
                f"Postgres did not accept SQL connections at {dsn} within "
                f"{timeout_seconds}s: {last_error}"
            )
        sleep(poll_interval_s)


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
    call_docker(*docker_cmd, env=os.environ | {"POSTGRES_PASSWORD": db_password})


def call_docker_start(container_name: str) -> None:
    args = ("start", container_name)
    result = run_or_error(*args, check_return=False)
    if result.returncode == 0 or "already running" in result.stderr.lower():
        return
    raise docker_error(args, result.stderr.strip())


def call_docker_stop(container_name: str) -> None:
    args = ("stop", container_name)
    result = run_or_error(*args, check_return=False)
    if result.returncode == 0 or "is not running" in result.stderr.lower():
        return
    raise docker_error(args, result.stderr.strip())


def call_docker_destroy(container_name: str, volume_name: str) -> None:
    remove_container = call_docker("rm", "-f", container_name, check=False)
    remove_volume = call_docker("volume", "rm", volume_name, check=False)
    if remove_container.returncode != 0:
        raise docker_error(
            ("rm", "-f", container_name), remove_container.stderr.strip()
        )
    if remove_volume.returncode != 0:
        raise docker_error(("volume", "rm", volume_name), remove_volume.stderr.strip())
