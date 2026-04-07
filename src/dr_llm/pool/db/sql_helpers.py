"""Shared SQL helpers for pool store modules."""

from __future__ import annotations

import json
import logging
from typing import Any, LiteralString, cast

from psycopg import errors as pg_errors
from psycopg import sql

from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.db.schema import PoolSchema

logger = logging.getLogger(__name__)


def q(query: str) -> sql.SQL:
    """Wrap a dynamically built query string as sql.SQL for psycopg's type checker.

    Safe because all interpolated identifiers (table/column names) are validated
    against ^[a-z][a-z0-9_]*$ by PoolSchema/KeyColumn before reaching here.
    """
    return sql.SQL(cast(LiteralString, query))


def is_constraint_error(exc: BaseException) -> bool:
    return isinstance(
        exc, (pg_errors.UniqueViolation, pg_errors.IntegrityConstraintViolation)
    )


def parse_json_field(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw or "{}")


def validate_key_values(schema: PoolSchema, key_values: dict[str, Any]) -> None:
    expected = set(schema.key_column_names)
    provided = set(key_values.keys())
    missing = expected - provided
    if missing:
        raise PoolSchemaError(f"Missing key columns: {missing}. Expected: {expected}")
    extra = provided - expected
    if extra:
        raise PoolSchemaError(f"Unexpected key columns: {extra}. Expected: {expected}")


def key_where_clause(
    schema: PoolSchema, key_values: dict[str, Any]
) -> tuple[str, list[Any]]:
    """Build WHERE clause for key column matching."""
    conditions: list[str] = []
    params: list[Any] = []
    for kc in schema.key_columns:
        conditions.append(f"{kc.name} = %s")
        params.append(key_values[kc.name])
    return " AND ".join(conditions), params


def key_values_from_row(schema: PoolSchema, row: dict[str, Any]) -> dict[str, Any]:
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
