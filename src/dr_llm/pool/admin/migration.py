"""Backfill ``pool_catalog`` rows for pools created before catalog persistence.

Pools created before the ``pool_catalog`` table existed have working sample
tables but no ``pool_catalog`` row, so :func:`dr_llm.pool.db.catalog.load_schema`
and friends raise ``UndefinedTable`` (the table itself is missing) or return
``None``. This module derives each pool's :class:`PoolSchema` by inspecting the
existing samples table and calls :meth:`PoolStore.ensure_schema` to create the
catalog table and persist the schema row. It is safe to re-run.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from sqlalchemy import text

from dr_llm.pool.admin.discovery import discover_pools_from_runtime
from dr_llm.pool.db import DbConfig, DbRuntime
from dr_llm.pool.db.catalog import load_schema
from dr_llm.pool.db.names import PoolTableType, SampleColumn
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema, pool_table_name
from dr_llm.pool.errors import PoolError
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.errors import ProjectError, ProjectNotFoundError
from dr_llm.project.project_info import ProjectInfo

logger = logging.getLogger(__name__)


class PoolMigrationError(PoolError):
    """Raised when pool catalog backfill cannot derive a schema."""


_SAMPLE_COLUMN_NAMES: frozenset[str] = frozenset(c.value for c in SampleColumn)

_SQL_TYPE_TO_COLUMN_TYPE: dict[str, ColumnType] = {
    "text": ColumnType.text,
    "character varying": ColumnType.text,
    "integer": ColumnType.integer,
    "bigint": ColumnType.integer,
    "smallint": ColumnType.integer,
    "boolean": ColumnType.boolean,
    "double precision": ColumnType.float_,
    "real": ColumnType.float_,
    "numeric": ColumnType.float_,
}

_COLUMNS_QUERY = text(
    "SELECT column_name, data_type "
    "FROM information_schema.columns "
    "WHERE table_schema = 'public' AND table_name = :table_name "
    "ORDER BY ordinal_position"
)


def derive_pool_schema(runtime: DbRuntime, pool_name: str) -> PoolSchema:
    """Reconstruct a :class:`PoolSchema` from an existing samples table.

    The schema's key columns are every column on ``pool_<name>_samples``
    that isn't a built-in :class:`SampleColumn`, in their stored ordinal order.
    Raises :class:`PoolMigrationError` if the samples table is missing or has
    a key column with an unrecognised SQL type.
    """
    samples_table_name = pool_table_name(pool_name, PoolTableType.SAMPLES)
    with runtime.connect() as conn:
        rows = conn.execute(
            _COLUMNS_QUERY, {"table_name": samples_table_name}
        ).fetchall()
    if not rows:
        raise PoolMigrationError(
            f"Cannot derive schema for pool {pool_name!r}: "
            f"samples table {samples_table_name!r} not found."
        )

    key_columns: list[KeyColumn] = []
    for column_name, data_type in rows:
        if column_name in _SAMPLE_COLUMN_NAMES:
            continue
        column_type = _SQL_TYPE_TO_COLUMN_TYPE.get(data_type)
        if column_type is None:
            raise PoolMigrationError(
                f"Cannot derive ColumnType for {column_name!r} on "
                f"{samples_table_name!r}: unsupported SQL type {data_type!r}."
            )
        key_columns.append(KeyColumn(name=column_name, type=column_type))

    if not key_columns:
        raise PoolMigrationError(
            f"Pool {pool_name!r} samples table has no key columns to derive."
        )

    return PoolSchema(name=pool_name, key_columns=key_columns)


class PoolBackfillStatus(StrEnum):
    backfilled = "backfilled"
    already_persisted = "already_persisted"
    would_backfill = "would_backfill"
    failed = "failed"


class PoolBackfillResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool_name: str
    status: PoolBackfillStatus
    derived_schema: PoolSchema | None = None
    message: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.status != PoolBackfillStatus.failed


class BackfillProjectCatalogRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    dry_run: bool = False

    @field_validator("project_name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return value.strip()


class BackfillProjectCatalogResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: BackfillProjectCatalogRequest
    project: ProjectInfo | None = None
    discovered_pool_names: list[str] = Field(default_factory=list)
    pool_results: list[PoolBackfillResult] = Field(default_factory=list)
    temporarily_started: bool = False
    message: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return all(result.success for result in self.pool_results)


def _backfill_one_pool(
    runtime: DbRuntime, pool_name: str, *, dry_run: bool
) -> PoolBackfillResult:
    try:
        existing = load_schema(runtime, pool_name)
    except Exception:
        # pool_catalog table does not exist yet; fall through and derive.
        existing = None

    if existing is not None:
        return PoolBackfillResult(
            pool_name=pool_name,
            status=PoolBackfillStatus.already_persisted,
            derived_schema=existing,
        )

    try:
        derived = derive_pool_schema(runtime, pool_name)
    except PoolMigrationError as exc:
        logger.exception("Failed to derive schema for pool %s", pool_name)
        return PoolBackfillResult(
            pool_name=pool_name, status=PoolBackfillStatus.failed, message=str(exc)
        )

    if dry_run:
        return PoolBackfillResult(
            pool_name=pool_name,
            status=PoolBackfillStatus.would_backfill,
            derived_schema=derived,
        )

    try:
        PoolStore(derived, runtime).ensure_schema()
    except Exception as exc:
        logger.exception("Failed to ensure schema for pool %s", pool_name)
        return PoolBackfillResult(
            pool_name=pool_name, status=PoolBackfillStatus.failed, message=str(exc)
        )

    return PoolBackfillResult(
        pool_name=pool_name,
        status=PoolBackfillStatus.backfilled,
        derived_schema=derived,
    )


def backfill_project_catalog(
    request: BackfillProjectCatalogRequest,
) -> BackfillProjectCatalogResult:
    """Discover pools in a project and persist their schemas to ``pool_catalog``.

    Starts the project if stopped, restoring the original state on exit. Each
    pool is processed independently — one pool's failure does not block the
    others. ``dry_run=True`` reports what would change without writing.
    """
    from dr_llm.project.project_service import (
        maybe_get_project,
        start_project,
        stop_project,
    )

    project = maybe_get_project(request.project_name)
    if project is None:
        raise ProjectNotFoundError(f"Project {request.project_name!r} not found")

    temporarily_started = False
    running_project = project
    discovered: list[str] = []
    results: list[PoolBackfillResult] = []
    try:
        if not project.running:
            running_project = start_project(request.project_name)
            temporarily_started = True
        if running_project.dsn is None:
            raise ProjectError(
                f"Project {running_project.name!r} has no DSN; start it first."
            )

        runtime = DbRuntime(DbConfig(dsn=running_project.dsn))
        try:
            discovered = discover_pools_from_runtime(runtime)
            results = [
                _backfill_one_pool(runtime, pool_name, dry_run=request.dry_run)
                for pool_name in discovered
            ]
        finally:
            runtime.close()
    finally:
        if temporarily_started:
            stop_project(request.project_name)

    if not discovered:
        message = f"Project {request.project_name!r} has no pools to backfill."
    elif request.dry_run:
        message = f"Dry run: inspected {len(discovered)} pool(s)."
    else:
        backfilled = sum(
            1 for r in results if r.status == PoolBackfillStatus.backfilled
        )
        skipped = sum(
            1 for r in results if r.status == PoolBackfillStatus.already_persisted
        )
        failed = sum(1 for r in results if r.status == PoolBackfillStatus.failed)
        message = (
            f"Backfilled {backfilled} pool(s), skipped {skipped} already-persisted, "
            f"failed {failed}."
        )

    return BackfillProjectCatalogResult(
        request=request,
        project=running_project,
        discovered_pool_names=discovered,
        pool_results=results,
        temporarily_started=temporarily_started,
        message=message,
    )
