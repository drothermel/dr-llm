from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy import Integer, MetaData, Table, Column, Text

from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.sql_helpers import (
    key_filter_clause,
    partial_key_filter_clause,
    validate_key_filter,
)
from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.key_filter import PoolKeyEqClause, PoolKeyFilter, PoolKeyInClause


def test_key_filter_clause_requires_exact_key_match() -> None:
    schema = PoolSchema(
        name="helpertest",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )
    table = Table(
        "pool_helpertest_samples",
        MetaData(),
        Column("dim_a", Text),
        Column("dim_b", Integer),
    )

    with pytest.raises(PoolSchemaError, match="Missing key columns"):
        key_filter_clause(schema, table, {"dim_a": "alpha"})
    with pytest.raises(PoolSchemaError, match="Unexpected key columns"):
        key_filter_clause(
            schema,
            table,
            {"dim_a": "alpha", "dim_b": 1, "extra": "nope"},
        )


def test_partial_key_filter_clause_supports_eq_and_in() -> None:
    schema = PoolSchema(
        name="helpertest",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )
    table = Table(
        "pool_helpertest_samples",
        MetaData(),
        Column("dim_a", Text),
        Column("dim_b", Integer),
    )

    clause = partial_key_filter_clause(
        schema,
        table,
        PoolKeyFilter(
            {
                "dim_a": PoolKeyInClause(values=["alpha", "beta"]),
                "dim_b": PoolKeyEqClause(value=3),
            }
        ),
    )

    assert clause is not None
    compiled = str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "dim_a IN ('alpha', 'beta')" in compiled
    assert "dim_b = 3" in compiled


def test_pool_key_filter_rejects_empty_in_values() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        PoolKeyFilter({"dim_a": PoolKeyInClause(values=[])})


def test_validate_key_filter_warns_on_unknown_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    schema = PoolSchema(
        name="helpertest",
        key_columns=[KeyColumn(name="dim_a")],
    )

    with caplog.at_level("WARNING", logger="dr_llm.pool.db.sql_helpers"):
        validate_key_filter(
            schema,
            PoolKeyFilter(
                {
                    "dim_a": PoolKeyEqClause(value="alpha"),
                    "extra": PoolKeyEqClause(value="nope"),
                }
            ),
        )

    assert any("unknown columns" in record.message for record in caplog.records)
