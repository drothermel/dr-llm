"""Unit tests for pool SQLAlchemy table metadata."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from dr_llm.pool.db import (
    ColumnType,
    KeyColumn,
    LeaseColumn,
    PoolSchema,
    PoolTableType,
    PoolTables,
    SampleColumn,
)
from dr_llm.pool.db.names import (
    IndexNamePrefix,
    PoolIndexName,
    pool_index_name,
)
from dr_llm.pool.db.schema import (
    pool_table_name,
    pool_table_names,
)
from dr_llm.pool.db.tables import LeasesTableDef, SamplesTableDef


def _simple_schema() -> PoolSchema:
    return PoolSchema(
        name="test",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )


def _index_by_name(tables: PoolTables, table_type: PoolTableType) -> dict[str, Index]:
    return {str(index.name): index for index in tables[table_type].indexes}


def _expression_names(index: Index) -> list[str]:
    return [str(expr).split(".")[-1] for expr in index.expressions]


def test_pool_tables_contains_new_tables_only() -> None:
    tables = PoolTables(_simple_schema())
    assert list(tables.tables) == list(PoolTableType)
    assert tables[PoolTableType.SAMPLES].name == "pool_test_samples"
    assert tables[PoolTableType.LEASES].name == "pool_test_leases"
    assert tables.all_tables == [tables[table_type] for table_type in PoolTableType]


def test_samples_table_def_builds_expected_columns() -> None:
    tables = PoolTables(_simple_schema())
    samples = tables[PoolTableType.SAMPLES]

    assert list(samples.c.keys()) == [
        "sample_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "run_id",
        "request_json",
        "response_json",
        "finish_reason",
        "attempt_count",
        "metadata_json",
        "created_at",
    ]
    assert samples.c.sample_id.primary_key
    assert not samples.c.sample_idx.nullable
    assert not samples.c.request_json.nullable
    assert samples.c.response_json.nullable
    assert not samples.c.attempt_count.nullable
    assert not samples.c.metadata_json.nullable
    assert not samples.c.created_at.nullable
    assert str(samples.c.dim_a.type) == "TEXT"
    assert str(samples.c.dim_b.type) == "INTEGER"
    assert isinstance(samples.c.request_json.type, JSONB)
    assert isinstance(samples.c.response_json.type, JSONB)
    assert isinstance(samples.c.metadata_json.type, JSONB)
    assert isinstance(samples.c.created_at.type, TIMESTAMP)
    assert samples.c.created_at.type.timezone is True


def test_samples_table_def_builds_expected_indexes() -> None:
    tables = PoolTables(_simple_schema())
    samples = tables[PoolTableType.SAMPLES]
    indexes = _index_by_name(tables, PoolTableType.SAMPLES)

    cell_index = indexes[
        pool_index_name(IndexNamePrefix.UNIQUE, samples.name, PoolIndexName.CELL)
    ]
    assert cell_index.unique is True
    assert _expression_names(cell_index) == [
        "dim_a",
        "dim_b",
        "sample_idx",
    ]

    key_index = indexes[
        pool_index_name(IndexNamePrefix.STANDARD, samples.name, PoolIndexName.KEY)
    ]
    assert _expression_names(key_index) == ["dim_a", "dim_b"]

    incomplete_index = indexes[
        pool_index_name(
            IndexNamePrefix.STANDARD,
            samples.name,
            PoolIndexName.INCOMPLETE,
        )
    ]
    assert _expression_names(incomplete_index) == ["created_at"]
    where_clause = incomplete_index.dialect_options["postgresql"]["where"]
    assert "response_json IS NULL" in str(where_clause)


def test_leases_table_def_builds_expected_columns() -> None:
    tables = PoolTables(_simple_schema())
    leases = tables[PoolTableType.LEASES]

    assert list(leases.c.keys()) == [
        "sample_id",
        "worker_id",
        "lease_expires_at",
    ]
    assert leases.c.sample_id.primary_key
    assert not leases.c.worker_id.nullable
    assert not leases.c.lease_expires_at.nullable
    assert isinstance(leases.c.lease_expires_at.type, TIMESTAMP)
    assert leases.c.lease_expires_at.type.timezone is True
    assert leases.indexes == set()


def test_pool_tables_key_columns_only_for_samples() -> None:
    tables = PoolTables(_simple_schema())

    assert [column.name for column in tables.key_columns(PoolTableType.SAMPLES)] == [
        "dim_a",
        "dim_b",
    ]
    with pytest.raises(ValueError, match="leases does not have pool key columns"):
        tables.key_columns(PoolTableType.LEASES)


def test_pool_tables_select_columns_are_table_type_specific() -> None:
    tables = PoolTables(_simple_schema())

    assert [column.name for column in tables.select_columns(PoolTableType.SAMPLES)] == [
        "sample_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "run_id",
        "request_json",
        "response_json",
        "finish_reason",
        "attempt_count",
        "metadata_json",
        "created_at",
    ]
    assert [column.name for column in tables.select_columns(PoolTableType.LEASES)] == [
        "sample_id",
        "worker_id",
        "lease_expires_at",
    ]


def test_pool_tables_registers_pydantic_table_defs() -> None:
    tables = PoolTables(_simple_schema())
    expected_def_types: dict[PoolTableType, type[BaseModel]] = {
        PoolTableType.SAMPLES: SamplesTableDef,
        PoolTableType.LEASES: LeasesTableDef,
    }

    assert list(tables.defs) == list(PoolTableType)
    for table_type, expected_type in expected_def_types.items():
        table_def = tables.defs[table_type]
        assert isinstance(table_def, BaseModel)
        assert isinstance(table_def, expected_type)
        assert table_def.table_type == table_type
        assert tables[table_type].name == _simple_schema().table_name(table_type)


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
    assert {table_type.value for table_type in PoolTableType} == {
        "samples",
        "leases",
    }
    assert pool_table_name("test", PoolTableType.SAMPLES) == "pool_test_samples"
    assert pool_table_name("test", PoolTableType.LEASES) == "pool_test_leases"
    assert schema.table_name(PoolTableType.SAMPLES) == "pool_test_samples"
    assert schema.table_name(PoolTableType.LEASES) == "pool_test_leases"
    assert schema.table_names() == pool_table_names("test")
    assert schema.table_names() == [
        "pool_test_samples",
        "pool_test_leases",
    ]


def test_column_enums() -> None:
    assert SampleColumn.REQUEST_JSON == "request_json"
    assert SampleColumn.RESPONSE_JSON == "response_json"
    assert LeaseColumn.LEASE_EXPIRES_AT == "lease_expires_at"


def test_pool_index_name() -> None:
    assert (
        pool_index_name(
            IndexNamePrefix.UNIQUE,
            "pool_test_samples",
            PoolIndexName.CELL,
        )
        == "uq_pool_test_samples_cell"
    )


def test_schema_key_column_names() -> None:
    schema = _simple_schema()
    assert schema.key_column_names == ["dim_a", "dim_b"]
