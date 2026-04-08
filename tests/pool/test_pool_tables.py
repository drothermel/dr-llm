"""Unit tests for pool SQLAlchemy table metadata."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects.postgresql import JSONB

from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.tables import PoolTables


def _simple_schema() -> PoolSchema:
    return PoolSchema(
        name="test",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )


def test_pool_tables_contains_all_tables() -> None:
    tables = PoolTables(_simple_schema())
    assert tables.samples.name == "pool_test_samples"
    assert tables.claims.name == "pool_test_claims"
    assert tables.pending.name == "pool_test_pending"
    assert tables.metadata_table.name == "pool_test_metadata"


def test_pool_tables_key_columns_appear() -> None:
    tables = PoolTables(_simple_schema())
    assert str(tables.samples.c.dim_a.type) == "TEXT"
    assert str(tables.samples.c.dim_b.type) == "INTEGER"
    assert str(tables.pending.c.dim_a.type) == "TEXT"
    assert str(tables.pending.c.dim_b.type) == "INTEGER"


def test_pool_tables_unique_index_includes_keys() -> None:
    tables = PoolTables(_simple_schema())
    sample_indexes = {
        index.name: [str(expr).split(".")[-1] for expr in index.expressions]
        for index in tables.samples.indexes
    }
    assert sample_indexes["uq_pool_test_samples_cell"] == [
        "dim_a",
        "dim_b",
        "sample_idx",
    ]


def test_pool_tables_json_columns_are_jsonb() -> None:
    tables = PoolTables(_simple_schema())
    assert isinstance(tables.samples.c.payload_json.type, JSONB)
    assert isinstance(tables.samples.c.metadata_json.type, JSONB)
    assert isinstance(tables.pending.c.payload_json.type, JSONB)
    assert isinstance(tables.pending.c.metadata_json.type, JSONB)
    assert isinstance(tables.metadata_table.c.value_json.type, JSONB)


def test_schema_name_validation() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        PoolSchema(name="Bad-Name", key_columns=[KeyColumn(name="x")])


def test_key_column_name_validation() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        KeyColumn(name="BadCol")


def test_schema_empty_key_columns_rejected() -> None:
    with pytest.raises(ValueError, match="at least one KeyColumn"):
        PoolSchema(name="empty", key_columns=[])


def test_schema_table_names() -> None:
    schema = _simple_schema()
    assert schema.samples_table == "pool_test_samples"
    assert schema.claims_table == "pool_test_claims"
    assert schema.pending_table == "pool_test_pending"
    assert schema.metadata_table == "pool_test_metadata"


def test_schema_key_column_names() -> None:
    schema = _simple_schema()
    assert schema.key_column_names == ["dim_a", "dim_b"]
