from __future__ import annotations

import pytest
from sqlalchemy import Integer, MetaData, Table, Column, Text

from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.sql_helpers import key_filter_clause
from dr_llm.pool.errors import PoolSchemaError


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

