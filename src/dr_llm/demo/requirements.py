"""Preflight checks shared by demo scripts."""

from __future__ import annotations

import typer

from dr_llm.demo.console import fail
from dr_llm.project.docker_runner import call_docker
from dr_llm.project.errors import DockerCommandError, DockerUnavailableError


def ensure_docker_available(
    *,
    reason: str,
    recovery_hint: str | None = None,
) -> None:
    """Exit with a clear demo message unless Docker is usable."""
    try:
        call_docker("version", "--format", "{{.Server.Version}}")
    except (DockerUnavailableError, DockerCommandError) as exc:
        message = f"Docker is required but unavailable.\n  {reason}"
        if recovery_hint is not None:
            message = f"{message}\n  {recovery_hint}"
        message = f"{message}\n  Detail: {exc}"
        fail(message)
        raise typer.Exit(1) from exc


__all__ = ["ensure_docker_available"]
