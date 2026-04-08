from __future__ import annotations

import json

from dr_llm.project.docker_project_metadata import DockerProjectMetadata
from dr_llm.project.docker_runner import _docker_error, _run_or_error
from dr_llm.project.errors import DockerContainerNotFoundError


def parse_docker_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for pair in raw.strip().split(","):
        k, _, v = pair.partition("=")
        labels[k.strip()] = v.strip()
    return labels


def get_docker_project_metadata(
    container_name: str,
) -> DockerProjectMetadata | None:
    args = (
        "inspect",
        "--format",
        DockerProjectMetadata.inspect_format(),
        container_name,
    )
    result = _run_or_error(*args, check_return=False)
    if result.returncode != 0:
        error = _docker_error(args, result.stderr.strip())
        if isinstance(error, DockerContainerNotFoundError):
            return None
        raise error
    return DockerProjectMetadata.from_inspect_output(result.stdout.strip())


def call_docker_list_labels() -> str:
    args = (
        "ps",
        "-a",
        "--filter",
        f"label={DockerProjectMetadata.name_label_key}",
        "--format",
        "{{json .}}",
    )
    return _run_or_error(*args).stdout.strip()


def get_all_docker_project_metadata() -> list[DockerProjectMetadata]:
    project_metadata: list[DockerProjectMetadata] = []
    name_key = DockerProjectMetadata.name_label_key
    for line in call_docker_list_labels().splitlines():
        if not line:
            continue
        data = json.loads(line)
        status = data.get("State")
        parsed = parse_docker_labels(data.get("Labels", ""))
        if name_key not in parsed:
            continue
        project_metadata.append(
            DockerProjectMetadata.from_labels_status(
                labels=parsed,
                status=status,
            )
        )
    return project_metadata
