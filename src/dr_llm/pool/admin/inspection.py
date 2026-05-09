from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from sqlalchemy import Column, DateTime, MetaData, Table, Text, select

from dr_llm.pool.db import (
    DbConfig,
    DbRuntime,
    MetadataColumn,
    PoolSchema,
    PoolTableType,
)
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pool_store import SCHEMA_METADATA_KEY
from dr_llm.pool.reader import PoolReader, _load_schema_from_db as load_schema_from_db
from dr_llm.project.errors import ProjectError, ProjectNotFoundError
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
    sample_count: int = 0
    pending_counts: PendingStatusCounts = Field(default_factory=PendingStatusCounts)

    @computed_field
    @property
    def pending_total(self) -> int:
        return self.pending_counts.total

    @computed_field
    @property
    def in_flight(self) -> int:
        return self.pending_counts.in_flight


def inspect_pool(request: PoolInspectionRequest) -> PoolInspection:
    from dr_llm.project.project_service import maybe_get_project

    project = maybe_get_project(request.project_name)
    if project is None:
        raise ProjectNotFoundError(f"Project {request.project_name!r} not found")
    return _inspect_pool_for_project(project, request.pool_name)


def _inspect_pool_for_project(project: ProjectInfo, pool_name: str) -> PoolInspection:
    if project.dsn is None:
        raise ProjectError(f"Project {project.name!r} has no DSN; start it first.")

    runtime = DbRuntime(DbConfig(dsn=project.dsn))
    try:
        schema = load_schema_from_db(runtime, pool_name)
        reader = PoolReader.from_runtime(runtime, schema=schema)
        progress = reader.progress()
        metadata_table = Table(
            schema.table_name(PoolTableType.METADATA),
            MetaData(),
            Column(MetadataColumn.POOL_NAME, Text, nullable=False),
            Column(MetadataColumn.KEY, Text, nullable=False),
            Column(MetadataColumn.CREATED_AT, DateTime(timezone=True)),
        )
        metadata_created_at_stmt = select(metadata_table.c.created_at).where(
            metadata_table.c.pool_name == schema.name,
            metadata_table.c.key == SCHEMA_METADATA_KEY,
        )
        with runtime.connect() as conn:
            created_at = conn.execute(metadata_created_at_stmt).scalar_one_or_none()
    finally:
        runtime.close()

    return PoolInspection(
        project_name=project.name,
        name=schema.name,
        pool_schema=schema,
        created_at=created_at,
        sample_count=progress.samples_total,
        pending_counts=progress.pending_counts,
    )
