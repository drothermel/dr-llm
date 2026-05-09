from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from dr_llm.datetime_utils import UTC, normalize_utc
from dr_llm.pool.admin.discovery import discover_pools
from dr_llm.pool.admin.inspection import (
    PoolInspection,
    PoolInspectionRequest,
    _inspect_pool_for_project,
    inspect_pool,
)
from dr_llm.pool.db import DbConfig, DbRuntime, KeyColumn, PoolSchema
from dr_llm.pool.db.schema import _VALID_NAME_RE
from dr_llm.pool.errors import PoolError
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.errors import ProjectError, ProjectNotFoundError
from dr_llm.project.project_info import ProjectInfo


class CreatePoolRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    key_axes: list[str] = Field(default_factory=list)

    @field_validator("project_name", "pool_name")
    @classmethod
    def _normalize_names(cls, value: str) -> str:
        return value.strip()

    @field_validator("key_axes")
    @classmethod
    def _normalize_key_axes(cls, value: list[str]) -> list[str]:
        return [axis.strip() for axis in value if axis.strip()]

    @classmethod
    def from_csv(
        cls, *, project_name: str, pool_name: str, axes_csv: str
    ) -> CreatePoolRequest:
        return cls(
            project_name=project_name,
            pool_name=pool_name,
            key_axes=[axis.strip() for axis in axes_csv.split(",") if axis.strip()],
        )

    @computed_field
    @property
    def has_key_axes(self) -> bool:
        return bool(self.key_axes)

    @computed_field
    @property
    def pool_name_is_valid(self) -> bool:
        return bool(_VALID_NAME_RE.match(self.pool_name))


class PoolCreationBlockReason(StrEnum):
    invalid_pool_name = "invalid_pool_name"
    missing_key_axes = "missing_key_axes"
    invalid_key_axis = "invalid_key_axis"
    project_not_found = "project_not_found"
    project_not_running = "project_not_running"
    pool_already_exists = "pool_already_exists"
    max_pools_reached = "max_pools_reached"
    cooldown_active = "cooldown_active"


class PoolCreationViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: PoolCreationBlockReason
    message: str
    project_name: str | None = None
    pool_name: str | None = None


class PoolCreationReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: CreatePoolRequest
    project: ProjectInfo | None = None
    existing_pools: list[PoolInspection] = Field(default_factory=list)
    violations: list[PoolCreationViolation] = Field(default_factory=list)

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


def assess_pool_creation(
    request: CreatePoolRequest,
    *,
    max_pools_per_project: int = 5,
    cooldown_seconds: int = 60,
) -> PoolCreationReadiness:
    from dr_llm.project.project_service import maybe_get_project

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
