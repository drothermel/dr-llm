"""Unit tests for pool SQLAlchemy table metadata."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB

from dr_llm.pool.db import ColumnType, KeyColumn, PoolSchema, PoolTableType, PoolTables
from dr_llm.pool.db.names import (
    IndexNamePrefix,
    PoolIndexName,
    pool_index_name,
)
from dr_llm.pool.db.schema import (
    pool_table_name,
    pool_table_names,
)
from dr_llm.pool.db.tables import (
    CallStatsTableDef,
    ClaimsTableDef,
    MetadataTableDef,
    PendingTableDef,
    SamplesTableDef,
)


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
    assert list(tables.tables) == list(PoolTableType)
    assert tables[PoolTableType.SAMPLES].name == "pool_test_samples"
    assert tables[PoolTableType.CLAIMS].name == "pool_test_claims"
    assert tables[PoolTableType.PENDING].name == "pool_test_pending"
    assert tables[PoolTableType.METADATA].name == "pool_test_metadata"
    assert tables[PoolTableType.CALL_STATS].name == "pool_test_call_stats"
    assert tables.all_tables == [tables[table_type] for table_type in PoolTableType]


def test_pool_tables_key_columns_appear() -> None:
    tables = PoolTables(_simple_schema())
    assert str(tables[PoolTableType.SAMPLES].c.dim_a.type) == "TEXT"
    assert str(tables[PoolTableType.SAMPLES].c.dim_b.type) == "INTEGER"
    assert str(tables[PoolTableType.PENDING].c.dim_a.type) == "TEXT"
    assert str(tables[PoolTableType.PENDING].c.dim_b.type) == "INTEGER"
    assert [column.name for column in tables.key_columns(PoolTableType.SAMPLES)] == [
        "dim_a",
        "dim_b",
    ]
    assert [column.name for column in tables.key_columns(PoolTableType.PENDING)] == [
        "dim_a",
        "dim_b",
    ]


def test_pool_tables_unique_index_includes_keys() -> None:
    tables = PoolTables(_simple_schema())
    sample_indexes = {
        str(index.name): [str(expr).split(".")[-1] for expr in index.expressions]
        for index in tables[PoolTableType.SAMPLES].indexes
    }
    index_name = pool_index_name(
        IndexNamePrefix.UNIQUE,
        tables[PoolTableType.SAMPLES].name,
        PoolIndexName.CELL,
    )
    assert sample_indexes[index_name] == [
        "dim_a",
        "dim_b",
        "sample_idx",
    ]


def test_pool_tables_json_columns_are_jsonb() -> None:
    tables = PoolTables(_simple_schema())
    assert isinstance(tables[PoolTableType.SAMPLES].c.payload_json.type, JSONB)
    assert isinstance(tables[PoolTableType.SAMPLES].c.metadata_json.type, JSONB)
    assert isinstance(tables[PoolTableType.PENDING].c.payload_json.type, JSONB)
    assert isinstance(tables[PoolTableType.PENDING].c.metadata_json.type, JSONB)
    assert isinstance(tables[PoolTableType.METADATA].c.value_json.type, JSONB)


def test_pool_tables_select_columns_are_table_type_specific() -> None:
    tables = PoolTables(_simple_schema())

    assert [column.name for column in tables.select_columns(PoolTableType.SAMPLES)] == [
        "sample_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "payload_json",
        "source_run_id",
        "metadata_json",
        "created_at",
    ]
    assert [column.name for column in tables.select_columns(PoolTableType.PENDING)] == [
        "pending_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "payload_json",
        "source_run_id",
        "metadata_json",
        "priority",
        "status",
        "worker_id",
        "lease_expires_at",
        "attempt_count",
        "created_at",
    ]
    for table_type in (
        PoolTableType.CLAIMS,
        PoolTableType.METADATA,
        PoolTableType.CALL_STATS,
    ):
        assert [column.name for column in tables.select_columns(table_type)] == list(
            tables[table_type].c.keys()
        )


def test_pool_tables_reject_missing_table_type_helpers() -> None:
    tables = PoolTables(_simple_schema())
    with pytest.raises(ValueError, match="key columns"):
        tables.key_columns(PoolTableType.CLAIMS)


def test_pool_tables_registers_pydantic_table_defs() -> None:
    tables = PoolTables(_simple_schema())
    expected_def_types: dict[PoolTableType, type[BaseModel]] = {
        PoolTableType.SAMPLES: SamplesTableDef,
        PoolTableType.CLAIMS: ClaimsTableDef,
        PoolTableType.PENDING: PendingTableDef,
        PoolTableType.METADATA: MetadataTableDef,
        PoolTableType.CALL_STATS: CallStatsTableDef,
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
    assert list(PoolTableType) == [
        "samples",
        "claims",
        "pending",
        "metadata",
        "call_stats",
    ]
    assert pool_table_name("test", PoolTableType.SAMPLES) == "pool_test_samples"
    assert schema.table_name(PoolTableType.SAMPLES) == "pool_test_samples"
    assert schema.table_name(PoolTableType.CLAIMS) == "pool_test_claims"
    assert schema.table_name(PoolTableType.PENDING) == "pool_test_pending"
    assert schema.table_name(PoolTableType.METADATA) == "pool_test_metadata"
    assert schema.table_name(PoolTableType.CALL_STATS) == "pool_test_call_stats"
    assert schema.table_names() == pool_table_names("test")
    assert schema.table_names() == [
        "pool_test_samples",
        "pool_test_claims",
        "pool_test_pending",
        "pool_test_metadata",
        "pool_test_call_stats",
    ]


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
