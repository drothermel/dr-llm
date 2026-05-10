"""Project lifecycle helpers shared by demo scripts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from pydantic import BaseModel

from dr_llm.demo.requirements import ensure_docker_available
from dr_llm.project import (
    CreateProjectRequest,
    ProjectInfo,
    create_project,
    destroy_project,
    maybe_get_project,
)


class DemoDsnLease(BaseModel):
    """A DSN plus any demo project cleanup responsibility."""

    dsn: str
    project_name: str | None = None
    should_destroy_project: bool = False


def create_demo_project(
    project_name: str,
    *,
    replace_existing: bool = False,
) -> ProjectInfo:
    """Create a demo project, optionally replacing an existing one first."""
    if replace_existing and maybe_get_project(project_name) is not None:
        destroy_project(project_name)
    return create_project(CreateProjectRequest(project_name=project_name))


def temporary_demo_project_name(prefix: str) -> str:
    """Return a unique short-lived demo project name."""
    return f"{prefix}_{uuid4().hex[:8]}"


def require_demo_project_dsn(project: ProjectInfo) -> str:
    """Return a demo project's DSN, raising if project creation omitted it."""
    if project.dsn is None:
        raise RuntimeError(f"Demo project {project.name!r} has no DSN.")
    return project.dsn


def prepare_demo_dsn(
    *,
    dsn: str | None,
    project_prefix: str,
    project_name: str | None = None,
    keep_project: bool = False,
    docker_reason: str,
    docker_recovery_hint: str | None = None,
) -> DemoDsnLease:
    """Return a usable DSN, creating a Docker demo project when needed."""
    if dsn is not None:
        return DemoDsnLease(dsn=dsn)

    ensure_docker_available(
        reason=docker_reason,
        recovery_hint=docker_recovery_hint,
    )
    resolved_project_name = project_name or temporary_demo_project_name(
        project_prefix
    )
    project = create_demo_project(resolved_project_name)
    try:
        resolved_dsn = require_demo_project_dsn(project)
    except Exception:
        if not keep_project:
            destroy_project(resolved_project_name)
        raise
    return DemoDsnLease(
        dsn=resolved_dsn,
        project_name=resolved_project_name,
        should_destroy_project=not keep_project,
    )


def cleanup_demo_dsn(lease: DemoDsnLease) -> None:
    """Destroy a temporary demo project when the lease owns cleanup."""
    if lease.project_name is None or not lease.should_destroy_project:
        return
    destroy_project(lease.project_name)


@contextmanager
def temporary_demo_project(project_name: str) -> Iterator[ProjectInfo]:
    """Create a demo project and always destroy it after use."""
    project: ProjectInfo | None = None
    try:
        project = create_demo_project(project_name)
        yield project
    finally:
        if project is not None:
            destroy_project(project_name)


__all__ = [
    "cleanup_demo_dsn",
    "create_demo_project",
    "DemoDsnLease",
    "prepare_demo_dsn",
    "require_demo_project_dsn",
    "temporary_demo_project",
    "temporary_demo_project_name",
]
