from __future__ import annotations

from dr_llm.project.docker_project_metadata import DockerProjectMetadata
from dr_llm.project.docker_runner import docker_error, run_or_error
from dr_llm.project.errors import DockerContainerNotFoundError


def get_docker_project_metadata(
    container_name: str,
) -> DockerProjectMetadata | None:
    args = (
        "inspect",
        "--format",
        DockerProjectMetadata.inspect_format(),
        container_name,
    )
    result = run_or_error(*args, check_return=False)
    if result.returncode != 0:
        error = docker_error(args, result.stderr.strip())
        if isinstance(error, DockerContainerNotFoundError):
            return None
        raise error
    return DockerProjectMetadata.from_inspect_output(result.stdout.strip())


def _list_project_container_names() -> list[str]:
    args = (
        "ps",
        "-a",
        "--filter",
        f"label={DockerProjectMetadata.name_label_key}",
        "--format",
        "{{.Names}}",
    )
    return [line for line in run_or_error(*args).stdout.splitlines() if line]


def get_all_docker_project_metadata() -> list[DockerProjectMetadata]:
    return [
        metadata
        for name in _list_project_container_names()
        if (metadata := get_docker_project_metadata(name)) is not None
    ]
