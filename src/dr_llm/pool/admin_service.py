from __future__ import annotations

import re
from datetime import datetime, timedelta

from sqlalchemy import Column, DateTime, MetaData, Table, Text, select, text

from dr_llm.datetime_utils import UTC, normalize_utc
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import (
    CreatePoolRequest,
    PoolCreationBlockReason,
    PoolCreationReadiness,
    PoolCreationViolation,
    PoolInspection,
    PoolInspectionRequest,
    PoolInspectionStatus,
)
from dr_llm.pool.errors import PoolError
from dr_llm.pool.pool_store import PoolStore, SCHEMA_METADATA_KEY
from dr_llm.pool.reader import PoolReader, _load_schema_from_db as load_schema_from_db
from dr_llm.project.errors import ProjectError, ProjectNotFoundError
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project.project_service import maybe_get_project

POOL_TABLE_RE = re.compile(r"^pool_(.+)_samples$")
POOL_DISCOVERY_SQL = text(
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = 'public' "
    r"AND table_name LIKE 'pool\_%\_samples' "
    "ORDER BY table_name"
)


def discover_pools(dsn: str) -> list[str]:
    runtime = DbRuntime(DbConfig(dsn=dsn))
    try:
        return discover_pools_from_runtime(runtime)
    finally:
        runtime.close()


def discover_pools_from_runtime(runtime: DbRuntime) -> list[str]:
    with runtime.connect() as conn:
        rows = conn.execute(POOL_DISCOVERY_SQL).fetchall()
    return [
        match.group(1)
        for (table_name,) in rows
        if (match := POOL_TABLE_RE.match(table_name))
    ]


def inspect_pool(request: PoolInspectionRequest) -> PoolInspection:
    project = maybe_get_project(request.project_name)
    if project is None:
        raise ProjectNotFoundError(f"Project {request.project_name!r} not found")
    return _inspect_pool_for_project(project, request.pool_name)


def assess_pool_creation(
    request: CreatePoolRequest,
    *,
    max_pools_per_project: int = 5,
    cooldown_seconds: int = 60,
) -> PoolCreationReadiness:
    violations = _request_violations(request)
    project = maybe_get_project(request.project_name)
    if project is None:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.project_not_found,
                message=f"Project {request.project_name!r} not found",
                project_name=request.project_name,
                pool_name=request.pool_name,
            )
        )
        return PoolCreationReadiness(request=request, violations=violations)

    if not project.running or project.dsn is None:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.project_not_running,
                message=(
                    f"Project {project.name!r} must be running before creating a pool."
                ),
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )
        return PoolCreationReadiness(
            request=request,
            project=project,
            violations=violations,
        )

    pool_names = discover_pools(project.dsn)
    existing_pools = [
        _inspect_pool_for_project(project, pool_name) for pool_name in pool_names
    ]

    if any(pool.name == request.pool_name for pool in existing_pools):
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.pool_already_exists,
                message=f"Pool {request.pool_name!r} already exists in project {project.name!r}.",
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )

    if len(existing_pools) >= max_pools_per_project:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.max_pools_reached,
                message=(
                    f"Project {project.name!r} already has {len(existing_pools)} pools; "
                    f"max_pools_per_project={max_pools_per_project}."
                ),
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )

    in_progress_pools = [
        pool.name
        for pool in existing_pools
        if pool.status == PoolInspectionStatus.in_progress
    ]
    if in_progress_pools:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.pool_in_progress,
                message=(
                    "Cannot create a new pool while other pools are in progress: "
                    + ", ".join(in_progress_pools)
                ),
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )

    cutoff = datetime.now(UTC) - timedelta(seconds=cooldown_seconds)
    recent_pools = [
        pool.name
        for pool in existing_pools
        if (created_at := normalize_utc(pool.created_at)) is not None
        and created_at >= cutoff
    ]
    if recent_pools:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.cooldown_active,
                message=(
                    "Cannot create a new pool yet; recent pools are still within the "
                    "cooldown window: " + ", ".join(recent_pools)
                ),
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )

    return PoolCreationReadiness(
        request=request,
        project=project,
        existing_pools=existing_pools,
        violations=violations,
    )


def create_pool(request: CreatePoolRequest) -> PoolInspection:
    readiness = assess_pool_creation(request)
    if not readiness.allowed:
        if any(
            violation.reason == PoolCreationBlockReason.project_not_found
            for violation in readiness.violations
        ):
            raise ProjectNotFoundError(f"Project {request.project_name!r} not found")
        assert readiness.blocked_message is not None
        if any(
            violation.reason == PoolCreationBlockReason.project_not_running
            for violation in readiness.violations
        ):
            raise ProjectError(readiness.blocked_message)
        raise PoolError(readiness.blocked_message)

    assert readiness.project is not None
    assert readiness.project.dsn is not None

    schema = PoolSchema.from_axis_names(request.pool_name, request.key_axes)
    runtime = DbRuntime(DbConfig(dsn=readiness.project.dsn))
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()
    finally:
        runtime.close()

    return inspect_pool(
        PoolInspectionRequest(
            project_name=request.project_name,
            pool_name=request.pool_name,
        )
    )


def _inspect_pool_for_project(project: ProjectInfo, pool_name: str) -> PoolInspection:
    if project.dsn is None:
        raise ProjectError(f"Project {project.name!r} has no DSN; start it first.")

    runtime = DbRuntime(DbConfig(dsn=project.dsn))
    try:
        schema = load_schema_from_db(runtime, pool_name)
        reader = PoolReader.from_runtime(runtime, schema=schema)
        progress = reader.progress()
        metadata_table = Table(
            schema.metadata_table,
            MetaData(),
            Column("pool_name", Text, nullable=False),
            Column("key", Text, nullable=False),
            Column("created_at", DateTime(timezone=True)),
        )
        metadata_created_at_stmt = select(metadata_table.c.created_at).where(
            metadata_table.c.pool_name == schema.name,
            metadata_table.c.key == SCHEMA_METADATA_KEY,
        )
        with runtime.connect() as conn:
            created_at = conn.execute(metadata_created_at_stmt).scalar_one_or_none()
    finally:
        runtime.close()

    if progress.samples_total == 0 and progress.pending_counts.total == 0:
        status = PoolInspectionStatus.empty
    elif progress.pending_counts.in_flight > 0:
        status = PoolInspectionStatus.in_progress
    else:
        status = PoolInspectionStatus.complete

    return PoolInspection(
        project_name=project.name,
        name=schema.name,
        pool_schema=schema,
        created_at=created_at,
        sample_count=progress.samples_total,
        pending_counts=progress.pending_counts,
        status=status,
    )


def _request_violations(request: CreatePoolRequest) -> list[PoolCreationViolation]:
    violations: list[PoolCreationViolation] = []
    if not request.pool_name_is_valid:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.invalid_pool_name,
                message=(
                    "pool_name must be lowercase alphanumeric with underscores, "
                    f"starting with a letter; got {request.pool_name!r}"
                ),
                project_name=request.project_name,
                pool_name=request.pool_name,
            )
        )

    if not request.has_key_axes:
        violations.append(
            PoolCreationViolation(
                reason=PoolCreationBlockReason.missing_key_axes,
                message="At least one key axis is required",
                project_name=request.project_name,
                pool_name=request.pool_name,
            )
        )
        return violations

    for axis in request.key_axes:
        try:
            KeyColumn(name=axis)
        except ValueError as exc:
            violations.append(
                PoolCreationViolation(
                    reason=PoolCreationBlockReason.invalid_key_axis,
                    message=str(exc),
                    project_name=request.project_name,
                    pool_name=request.pool_name,
                )
            )
    return violations
