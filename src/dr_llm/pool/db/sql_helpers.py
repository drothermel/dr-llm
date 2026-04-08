"""Shared SQL helpers for pool store modules."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from psycopg import errors as pg_errors
from sqlalchemy import and_
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.db.schema import PoolSchema

logger = logging.getLogger(__name__)


def is_constraint_error(exc: BaseException) -> bool:
    if isinstance(exc, IntegrityError):
        orig = exc.orig
        return isinstance(
            orig, (pg_errors.UniqueViolation, pg_errors.IntegrityConstraintViolation)
        )
    if isinstance(exc, DBAPIError) and exc.orig is not None:
        return is_constraint_error(exc.orig)
    return isinstance(
        exc, (pg_errors.UniqueViolation, pg_errors.IntegrityConstraintViolation)
    )


def parse_json_field(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in parse_json_field: {raw!r}") from err


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


def resolve_group_column(schema: PoolSchema, table: Table, group_column: str) -> Any:
    if group_column not in schema.key_column_names:
        raise PoolSchemaError(f"group_column {group_column!r} not in schema")
    return table.c[group_column]


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
