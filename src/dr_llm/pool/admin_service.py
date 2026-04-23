from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Final

from sqlalchemy import Column, DateTime, MetaData, Table, Text, select, text
from sqlalchemy.engine import Connection

from dr_llm.datetime_utils import UTC, normalize_utc
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import (
    CreatePoolRequest,
    DeletePoolRequest,
    DeletePoolsByTokenRequest,
    DeletePoolsByTokenResult,
    DeletePoolsByTokenStatus,
    PoolCreationBlockReason,
    PoolCreationReadiness,
    PoolCreationViolation,
    PoolDeletionBlockReason,
    PoolDeletionReadiness,
    PoolDeletionResult,
    PoolDeletionStatus,
    PoolDeletionViolation,
    PoolInspection,
    PoolInspectionRequest,
    PoolInspectionStatus,
)
from dr_llm.pool.errors import PoolError
from dr_llm.pool.pool_store import PoolStore, SCHEMA_METADATA_KEY
from dr_llm.pool.reader import PoolReader, _load_schema_from_db as load_schema_from_db
from dr_llm.project.docker_psql import validate_pg_identifier
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
_POOL_TABLE_SUFFIXES: Final[tuple[str, ...]] = (
    "samples",
    "claims",
    "pending",
    "metadata",
    "call_stats",
)
_IN_PROGRESS_PENDING_STATUSES: Final[tuple[str, ...]] = ("pending", "leased")
_DEFAULT_TESTISH_TOKENS: Final[frozenset[str]] = frozenset(
    {"test", "tst", "smoke", "demo"}
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


def assess_pool_deletion(request: DeletePoolRequest) -> PoolDeletionReadiness:
    violations = _pool_delete_request_violations(request)
    project = maybe_get_project(request.project_name)
    if project is None:
        violations.append(
            PoolDeletionViolation(
                reason=PoolDeletionBlockReason.project_not_found,
                message=f"Project {request.project_name!r} not found",
                project_name=request.project_name,
                pool_name=request.pool_name,
            )
        )
        return PoolDeletionReadiness(request=request, violations=violations)

    if not project.running or project.dsn is None:
        violations.append(
            PoolDeletionViolation(
                reason=PoolDeletionBlockReason.project_not_running,
                message=(
                    f"Project {project.name!r} must be running before deleting a pool."
                ),
                project_name=project.name,
                pool_name=request.pool_name,
            )
        )
        return PoolDeletionReadiness(
            request=request,
            project=project,
            violations=violations,
        )
    if violations:
        return PoolDeletionReadiness(
            request=request,
            project=project,
            violations=violations,
        )

    runtime = DbRuntime(DbConfig(dsn=project.dsn))
    try:
        existing_table_names = _existing_pool_table_names(runtime, request.pool_name)
        if not existing_table_names:
            violations.append(
                PoolDeletionViolation(
                    reason=PoolDeletionBlockReason.pool_not_found,
                    message=(
                        f"Pool {request.pool_name!r} does not exist in project "
                        f"{project.name!r}."
                    ),
                    project_name=project.name,
                    pool_name=request.pool_name,
                )
            )
            return PoolDeletionReadiness(
                request=request,
                project=project,
                existing_table_names=existing_table_names,
                violations=violations,
            )

        in_progress_pending_count = _count_in_progress_pending_rows(
            runtime, request.pool_name
        )
        if in_progress_pending_count > 0:
            violations.append(
                PoolDeletionViolation(
                    reason=PoolDeletionBlockReason.pool_in_progress,
                    message=(
                        f"Pool {request.pool_name!r} is still in progress and cannot "
                        "be deleted yet."
                    ),
                    project_name=project.name,
                    pool_name=request.pool_name,
                )
            )
        return PoolDeletionReadiness(
            request=request,
            project=project,
            existing_table_names=existing_table_names,
            in_progress_pending_count=in_progress_pending_count,
            violations=violations,
        )
    finally:
        runtime.close()


def delete_pool(request: DeletePoolRequest) -> PoolDeletionResult:
    readiness = assess_pool_deletion(request)
    target_table_names = _pool_table_names(request.pool_name)
    missing_table_names = [
        table_name
        for table_name in target_table_names
        if table_name not in readiness.existing_table_names
    ]
    if not readiness.allowed:
        return PoolDeletionResult(
            request=request,
            project=readiness.project,
            status=PoolDeletionStatus.blocked,
            existing_table_names=readiness.existing_table_names,
            missing_table_names=missing_table_names,
            violations=readiness.violations,
            message=readiness.blocked_message,
        )

    assert readiness.project is not None
    assert readiness.project.dsn is not None

    runtime = DbRuntime(DbConfig(dsn=readiness.project.dsn))
    pre_delete_counts: dict[str, int] = {}
    try:
        existing_table_names = _existing_pool_table_names(runtime, request.pool_name)
        missing_table_names = [
            table_name
            for table_name in target_table_names
            if table_name not in existing_table_names
        ]
        with runtime.begin() as conn:
            for table_name in existing_table_names:
                pre_delete_counts[table_name] = _table_row_count(conn, table_name)
            for table_name in existing_table_names:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
        remaining_table_names = _existing_pool_table_names(runtime, request.pool_name)
        status = (
            PoolDeletionStatus.deleted
            if not remaining_table_names
            else PoolDeletionStatus.failed
        )
        message = None
        if remaining_table_names:
            message = (
                f"Pool {request.pool_name!r} still has remaining tables after "
                f"deletion: {', '.join(remaining_table_names)}"
            )
        return PoolDeletionResult(
            request=request,
            project=readiness.project,
            status=status,
            existing_table_names=existing_table_names,
            deleted_table_names=existing_table_names,
            missing_table_names=missing_table_names,
            remaining_table_names=remaining_table_names,
            pre_delete_counts=pre_delete_counts,
            message=message,
        )
    except Exception as exc:
        return PoolDeletionResult(
            request=request,
            project=readiness.project,
            status=PoolDeletionStatus.failed,
            existing_table_names=readiness.existing_table_names,
            deleted_table_names=[],
            missing_table_names=missing_table_names,
            pre_delete_counts=pre_delete_counts,
            message=str(exc),
        )
    finally:
        runtime.close()


def delete_pools_by_token(
    request: DeletePoolsByTokenRequest,
) -> DeletePoolsByTokenResult:
    project = maybe_get_project(request.project_name)
    if project is None:
        return DeletePoolsByTokenResult(
            request=request,
            status=DeletePoolsByTokenStatus.blocked,
            dry_run=request.dry_run,
            message=f"Project {request.project_name!r} not found",
        )
    if not project.running or project.dsn is None:
        return DeletePoolsByTokenResult(
            request=request,
            project=project,
            status=DeletePoolsByTokenStatus.blocked,
            dry_run=request.dry_run,
            message=(
                f"Project {project.name!r} must be running before deleting matching pools."
            ),
        )

    discovered_pool_names = discover_pools(project.dsn)
    matched_pool_names = [
        pool_name
        for pool_name in discovered_pool_names
        if pool_name_has_token_match(pool_name, request.match_tokens)
    ]
    if request.dry_run:
        return DeletePoolsByTokenResult(
            request=request,
            project=project,
            status=DeletePoolsByTokenStatus.completed,
            discovered_pool_names=discovered_pool_names,
            matched_pool_names=matched_pool_names,
            dry_run=True,
            message=(
                "No matching pools found."
                if not matched_pool_names
                else f"Dry run: would delete {len(matched_pool_names)} matching pools."
            ),
        )
    pool_results = [
        delete_pool(
            DeletePoolRequest(project_name=request.project_name, pool_name=pool_name)
        )
        for pool_name in matched_pool_names
    ]
    if any(not result.success for result in pool_results):
        failed_pool_names = [
            result.request.pool_name for result in pool_results if not result.success
        ]
        return DeletePoolsByTokenResult(
            request=request,
            project=project,
            status=DeletePoolsByTokenStatus.failed,
            discovered_pool_names=discovered_pool_names,
            matched_pool_names=matched_pool_names,
            pool_results=pool_results,
            dry_run=False,
            message="Failed to delete matching pools: " + ", ".join(failed_pool_names),
        )
    return DeletePoolsByTokenResult(
        request=request,
        project=project,
        status=DeletePoolsByTokenStatus.completed,
        discovered_pool_names=discovered_pool_names,
        matched_pool_names=matched_pool_names,
        pool_results=pool_results,
        dry_run=False,
        message=(
            "No matching pools found."
            if not matched_pool_names
            else f"Deleted {len(matched_pool_names)} matching pools."
        ),
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


def _pool_delete_request_violations(
    request: DeletePoolRequest,
) -> list[PoolDeletionViolation]:
    violations: list[PoolDeletionViolation] = []
    if not request.pool_name_is_valid:
        violations.append(
            PoolDeletionViolation(
                reason=PoolDeletionBlockReason.invalid_pool_name,
                message=(
                    "pool_name must be lowercase alphanumeric with underscores, "
                    f"starting with a letter; got {request.pool_name!r}"
                ),
                project_name=request.project_name,
                pool_name=request.pool_name,
            )
        )
    return violations


def _pool_table_names(pool_name: str) -> list[str]:
    return [f"pool_{pool_name}_{suffix}" for suffix in _POOL_TABLE_SUFFIXES]


def pool_name_tokens(pool_name: str) -> list[str]:
    return [token.lower() for token in pool_name.split("_") if token]


def pool_name_has_token_match(
    pool_name: str,
    match_tokens: list[str] | None = None,
) -> bool:
    token_set = set(match_tokens or _DEFAULT_TESTISH_TOKENS)
    return any(token in token_set for token in pool_name_tokens(pool_name))


def _validated_pool_table_names(pool_name: str) -> list[str]:
    table_names = _pool_table_names(pool_name)
    for table_name in table_names:
        validate_pg_identifier(table_name, "table name")
    return table_names


def _existing_pool_table_names(runtime: DbRuntime, pool_name: str) -> list[str]:
    table_names = _validated_pool_table_names(pool_name)
    existing: list[str] = []
    with runtime.connect() as conn:
        for table_name in table_names:
            exists = conn.execute(
                text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = :table_name"
                    ")"
                ),
                {"table_name": table_name},
            ).scalar_one()
            if bool(exists):
                existing.append(table_name)
    return existing


def _count_in_progress_pending_rows(runtime: DbRuntime, pool_name: str) -> int:
    pending_table = f"pool_{pool_name}_pending"
    validate_pg_identifier(pending_table, "table name")
    existing_table_names = _existing_pool_table_names(runtime, pool_name)
    if pending_table not in existing_table_names:
        return 0
    placeholders = ", ".join(
        f":status_{idx}" for idx, _ in enumerate(_IN_PROGRESS_PENDING_STATUSES)
    )
    params = {
        f"status_{idx}": status
        for idx, status in enumerate(_IN_PROGRESS_PENDING_STATUSES)
    }
    with runtime.connect() as conn:
        return int(
            conn.execute(
                text(
                    f'SELECT count(*) FROM "{pending_table}" '
                    f"WHERE status IN ({placeholders})"
                ),
                params,
            ).scalar_one()
        )


def _table_row_count(conn: Connection, table_name: str) -> int:
    validate_pg_identifier(table_name, "table name")
    return int(conn.execute(text(f'SELECT count(*) FROM "{table_name}"')).scalar_one())


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
