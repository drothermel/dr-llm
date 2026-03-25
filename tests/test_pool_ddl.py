"""Unit tests for pool DDL generation."""

from __future__ import annotations

import pytest

from dr_llm.pool.ddl import generate_ddl
from dr_llm.pool.schema import ColumnType, KeyColumn, PoolSchema


def _simple_schema() -> PoolSchema:
    return PoolSchema(
        name="test",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )


def test_generate_ddl_contains_all_tables() -> None:
    ddl = generate_ddl(_simple_schema())
    assert "pool_test_samples" in ddl
    assert "pool_test_claims" in ddl
    assert "pool_test_pending" in ddl
    assert "pool_test_metadata" in ddl


def test_generate_ddl_key_columns_appear() -> None:
    ddl = generate_ddl(_simple_schema())
    assert "dim_a TEXT NOT NULL" in ddl
    assert "dim_b INTEGER NOT NULL" in ddl


def test_generate_ddl_unique_index_includes_keys() -> None:
    ddl = generate_ddl(_simple_schema())
    assert "dim_a, dim_b, sample_idx" in ddl


def test_generate_ddl_idempotent_syntax() -> None:
    ddl = generate_ddl(_simple_schema())
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS" in ddl
    assert "CREATE INDEX IF NOT EXISTS" in ddl


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
    s = _simple_schema()
    assert s.samples_table == "pool_test_samples"
    assert s.claims_table == "pool_test_claims"
    assert s.pending_table == "pool_test_pending"
    assert s.metadata_table == "pool_test_metadata"


def test_schema_key_column_names() -> None:
    s = _simple_schema()
    assert s.key_column_names == ["dim_a", "dim_b"]
