"""Shared SQL helpers for pool store modules."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from psycopg import errors as pg_errors
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import Insert as PgInsert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.errors import PoolSchemaError

logger = logging.getLogger(__name__)


def is_constraint_error(exc: BaseException) -> bool:
    orig = exc.orig if isinstance(exc, DBAPIError) else exc
    return isinstance(
        orig, (pg_errors.UniqueViolation, pg_errors.IntegrityConstraintViolation)
    )


def validate_key_values(schema: PoolSchema, key_values: dict[str, Any]) -> None:
    expected = set(schema.key_column_names)
    provided = set(key_values.keys())
    missing = expected - provided
    if missing:
        raise PoolSchemaError(f"Missing key columns: {missing}. Expected: {expected}")
    extra = provided - expected
    if extra:
        raise PoolSchemaError(f"Unexpected key columns: {extra}. Expected: {expected}")


def key_filter_clause(
    schema: PoolSchema,
    table: Table,
    key_values: Mapping[str, Any],
) -> ColumnElement[bool]:
    validate_key_values(schema, dict(key_values))
    return and_(*[table.c[kc.name] == key_values[kc.name] for kc in schema.key_columns])


def partial_key_filter_clause(
    schema: PoolSchema,
    table: Table,
    key_filter: Mapping[str, Any] | None,
) -> ColumnElement[bool] | None:
    if not key_filter:
        return None
    validate_key_filter(schema, dict(key_filter))
    conditions = [
        table.c[key] == value
        for key, value in key_filter.items()
        if key in schema.key_column_names
    ]
    if not conditions:
        return None
    return and_(*conditions)


def key_values_from_row(
    schema: PoolSchema, row: Mapping[str, Any] | RowMapping
) -> dict[str, Any]:
    return {name: row[name] for name in schema.key_column_names}


def validate_key_filter(schema: PoolSchema, key_filter: dict[str, Any]) -> None:
    """Warn on unknown keys in a partial key filter.

    Unlike validate_key_values (which requires an exact key match and raises
    PoolSchemaError), this is intentionally permissive: key_filter is a partial
    subset, so unknown keys are logged as warnings rather than errors.
    """
    unknown = set(key_filter.keys()) - set(schema.key_column_names)
    if unknown:
        logger.warning(
            "key_filter contains unknown columns %s (valid: %s)",
            unknown,
            schema.key_column_names,
        )


def insert_keyed_samples(
    runtime: DbRuntime,
    table: Table,
    pk_column: ColumnElement[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    ignore_conflicts: bool = True,
) -> int:
    """Insert ``rows`` into ``table`` and return the inserted row count.

    Builds ``pg_insert(table).returning(pk_column)`` once and dispatches to
    ``execute_insert_count``: a single row uses ``.values()`` for a tighter
    statement, multiple rows fan out via executemany. Conflict tolerance is
    delegated, so callers get the same race-safe behavior as
    ``execute_insert_count``.
    """
    if not rows:
        return 0
    stmt = pg_insert(table).returning(pk_column)
    if len(rows) == 1:
        return execute_insert_count(
            runtime,
            stmt.values(**rows[0]),
            ignore_conflicts=ignore_conflicts,
        )
    return execute_insert_count(
        runtime,
        stmt,
        ignore_conflicts=ignore_conflicts,
        parameters=list(rows),
    )


def execute_insert_count(
    runtime: DbRuntime,
    stmt: PgInsert[Any],
    *,
    ignore_conflicts: bool,
    parameters: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Execute INSERT (optionally ON CONFLICT DO NOTHING); return inserted row count.

    The statement must include a RETURNING clause so the inserted rows can be
    counted via ``.scalars()``. Pass ``parameters`` to use executemany semantics
    (one row per mapping) when bulk-inserting from a list of dicts.

    On a constraint violation with ``ignore_conflicts=True`` the helper returns
    0 instead of raising — covers the rare race where another writer raced past
    the ON CONFLICT guard.
    """
    if ignore_conflicts:
        stmt = stmt.on_conflict_do_nothing()
    with runtime.begin() as conn:
        try:
            result = (
                conn.execute(stmt)
                if parameters is None
                else conn.execute(stmt, parameters)
            )
            inserted = 0
            for _ in result.scalars():
                inserted += 1
            return inserted
        except Exception as exc:
            if ignore_conflicts and is_constraint_error(exc):
                return 0
            raise


def stream_select_rows(
    runtime: DbRuntime,
    schema: PoolSchema,
    table: Table,
    select_columns: Sequence[Any],
    *,
    base_predicates: Sequence[ColumnElement[bool]] = (),
    order_by: Sequence[Any] = (),
    key_filter: Mapping[str, Any] | None = None,
    chunk_size: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Stream rows from ``table`` in chunks via server-side cursoring.

    Builds ``SELECT select_columns FROM table WHERE base_predicates AND
    partial_key_filter ORDER BY order_by`` and yields each row as a dict.
    Uses SQLAlchemy's ``yield_per`` so the driver fetches ``chunk_size`` rows
    at a time. The underlying connection is held open for the lifetime of the
    iterator — fully consume or close it promptly.
    """
    stmt = select(*select_columns)
    if base_predicates:
        stmt = stmt.where(*base_predicates)
    key_clause = partial_key_filter_clause(schema, table, key_filter)
    if key_clause is not None:
        stmt = stmt.where(key_clause)
    if order_by:
        stmt = stmt.order_by(*order_by)

    with runtime.connect() as conn:
        result = conn.execution_options(yield_per=chunk_size).execute(stmt)
        for row in result.mappings():
            yield dict(row)
