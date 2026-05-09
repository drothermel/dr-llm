from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from sqlalchemy import text
from sqlalchemy.engine import Connection

from dr_llm.pool.admin.discovery import discover_pools, pool_name_has_token_match
from dr_llm.pool.db import DbConfig, DbRuntime, PendingColumn, PoolTableType
from dr_llm.pool.db.schema import _VALID_NAME_RE, pool_table_name, pool_table_names
from dr_llm.project.docker_psql import validate_pg_identifier
from dr_llm.project.project_info import ProjectInfo

_IN_PROGRESS_PENDING_STATUSES: Final[tuple[str, ...]] = ("pending", "leased")


class DeletePoolRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str

    @field_validator("project_name", "pool_name")
    @classmethod
    def _normalize_names(cls, value: str) -> str:
        return value.strip()

    @computed_field
    @property
    def pool_name_is_valid(self) -> bool:
        return bool(_VALID_NAME_RE.match(self.pool_name))


class PoolDeletionBlockReason(StrEnum):
    invalid_pool_name = "invalid_pool_name"
    project_not_found = "project_not_found"
    project_not_running = "project_not_running"
    pool_not_found = "pool_not_found"
    pool_in_progress = "pool_in_progress"


class PoolDeletionViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: PoolDeletionBlockReason
    message: str
    project_name: str | None = None
    pool_name: str | None = None


class PoolDeletionReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: DeletePoolRequest
    project: ProjectInfo | None = None
    existing_table_names: list[str] = Field(default_factory=list)
    in_progress_pending_count: int = 0
    violations: list[PoolDeletionViolation] = Field(default_factory=list)

    @computed_field
    @property
    def allowed(self) -> bool:
        return not self.violations

    @computed_field
    @property
    def blocked_message(self) -> str | None:
        if self.allowed:
            return None
        return "\n".join(violation.message for violation in self.violations)


class PoolDeletionStatus(StrEnum):
    deleted = "deleted"
    blocked = "blocked"
    failed = "failed"
    cancelled = "cancelled"


class PoolDeletionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: DeletePoolRequest
    project: ProjectInfo | None = None
    status: PoolDeletionStatus
    existing_table_names: list[str] = Field(default_factory=list)
    deleted_table_names: list[str] = Field(default_factory=list)
    missing_table_names: list[str] = Field(default_factory=list)
    remaining_table_names: list[str] = Field(default_factory=list)
    pre_delete_counts: dict[str, int] = Field(default_factory=dict)
    violations: list[PoolDeletionViolation] = Field(default_factory=list)
    message: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.status == PoolDeletionStatus.deleted


class DeletePoolsByTokenRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    match_tokens: list[str] = Field(default_factory=list)
    dry_run: bool = False

    @field_validator("project_name")
    @classmethod
    def _normalize_project_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("match_tokens")
    @classmethod
    def _normalize_match_tokens(cls, value: list[str]) -> list[str]:
        return [token.strip().lower() for token in value if token.strip()]


class DeletePoolsByTokenStatus(StrEnum):
    completed = "completed"
    blocked = "blocked"
    failed = "failed"


class DeletePoolsByTokenResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: DeletePoolsByTokenRequest
    project: Any | None = None
    status: DeletePoolsByTokenStatus
    discovered_pool_names: list[str] = Field(default_factory=list)
    matched_pool_names: list[str] = Field(default_factory=list)
    pool_results: list[PoolDeletionResult] = Field(default_factory=list)
    dry_run: bool = False
    message: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.status == DeletePoolsByTokenStatus.completed


def assess_pool_deletion(request: DeletePoolRequest) -> PoolDeletionReadiness:
    from dr_llm.project.project_service import maybe_get_project

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
    from dr_llm.project.project_service import maybe_get_project

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
    return pool_table_names(pool_name)


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
    pending_table = pool_table_name(pool_name, PoolTableType.PENDING)
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
                    f"WHERE {PendingColumn.STATUS} IN ({placeholders})"
                ),
                params,
            ).scalar_one()
        )


def _table_row_count(conn: Connection, table_name: str) -> int:
    validate_pg_identifier(table_name, "table name")
    return int(conn.execute(text(f'SELECT count(*) FROM "{table_name}"')).scalar_one())
