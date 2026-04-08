from __future__ import annotations

from time import sleep

from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
)
from dr_llm.project.docker_runner import (
    _docker_error,
    _run_or_error,
    _temp_environ,
    call_docker,
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
        result = _run_or_error(*args, check_return=False)
        if result.returncode == 0:
            return ContainerStatus.RUNNING
        error = _docker_error(args, result.stderr.strip())
        if isinstance(error, (DockerContainerNotFoundError, DockerUnavailableError)):
            raise error
        sleep(1)
    raise DockerCommandError(
        f"Postgres in {container_name} did not become ready within {timeout_seconds}s"
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
    args = ("start", container_name)
    result = _run_or_error(*args, check_return=False)
    if result.returncode == 0 or "already running" in result.stderr.lower():
        return
    raise _docker_error(args, result.stderr.strip())


def call_docker_stop(container_name: str) -> None:
    args = ("stop", container_name)
    result = _run_or_error(*args, check_return=False)
    if result.returncode == 0 or "is not running" in result.stderr.lower():
        return
    raise _docker_error(args, result.stderr.strip())


def call_docker_destroy(container_name: str, volume_name: str) -> None:
    remove_container = call_docker("rm", "-f", container_name, check=False)
    remove_volume = call_docker("volume", "rm", volume_name, check=False)
    if remove_container.returncode != 0:
        raise _docker_error(
            ("rm", "-f", container_name), remove_container.stderr.strip()
        )
    if remove_volume.returncode != 0:
        raise _docker_error(("volume", "rm", volume_name), remove_volume.stderr.strip())
