"""Project lifecycle helpers shared by demo scripts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from dr_llm.project import (
    CreateProjectRequest,
    ProjectInfo,
    create_project,
    destroy_project,
    maybe_get_project,
)


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
    "create_demo_project",
    "temporary_demo_project",
    "temporary_demo_project_name",
]
