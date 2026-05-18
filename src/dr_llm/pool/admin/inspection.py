"""Pool inspection: introspect pool state and schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from dr_llm.pool.db import DbConfig, DbRuntime, PoolSchema
from dr_llm.pool.db.catalog import load_catalog_created_at, load_schema
from dr_llm.pool.errors import PoolNotFoundError
from dr_llm.pool.pool_progress import PoolProgress
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.errors import ProjectNotFoundError
from dr_llm.project.project_info import ProjectInfo


class PoolInspectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str

    @field_validator("project_name", "pool_name")
    @classmethod
    def _normalize_names(cls, value: str) -> str:
        return value.strip()


class PoolInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    name: str
    pool_schema: PoolSchema
    created_at: datetime | None = None
    progress: PoolProgress


def inspect_pool_dsn(
    *, dsn: str, pool_name: str, project_name: str = "dsn"
) -> PoolInspection:
    return _inspect_pool_for_dsn(
        dsn=dsn,
        pool_name=pool_name.strip(),
        project_name=project_name.strip(),
    )


def inspect_pool(request: PoolInspectionRequest) -> PoolInspection:
    from dr_llm.project.project_service import maybe_get_project

    project = maybe_get_project(request.project_name)
    if project is None:
        raise ProjectNotFoundError(
            f"Project {request.project_name!r} not found"
        )
    return _inspect_pool_for_project(project, request.pool_name)


def _inspect_pool_for_dsn(
    *,
    dsn: str,
    pool_name: str,
    project_name: str,
) -> PoolInspection:
    runtime = DbRuntime(DbConfig(dsn=dsn))
    try:
        schema = load_schema(runtime, pool_name)
        if schema is None:
            raise PoolNotFoundError(
                f"Pool {pool_name!r} not found in the catalog."
            )
        store = PoolStore(schema, runtime)
        pool_progress = store.progress()
        created_at = load_catalog_created_at(runtime, pool_name)
    finally:
        runtime.close()

    return PoolInspection(
        project_name=project_name,
        name=schema.name,
        pool_schema=schema,
        created_at=created_at,
        progress=pool_progress,
    )


def _inspect_pool_for_project(
    project: ProjectInfo, pool_name: str
) -> PoolInspection:
    from dr_llm.project.errors import ProjectError

    if project.dsn is None:
        raise ProjectError(
            f"Project {project.name!r} has no DSN; start it first."
        )

    return _inspect_pool_for_dsn(
        dsn=project.dsn,
        pool_name=pool_name,
        project_name=project.name,
    )
