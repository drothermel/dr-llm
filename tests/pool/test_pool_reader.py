"""Unit tests for PoolReader construction and delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.errors import PoolNotFoundError
from dr_llm.pool.reader import PoolReader


_SCHEMA = PoolSchema(
    name=f"unit_reader_{uuid4().hex[:8]}",
    key_columns=[KeyColumn(name="dim_a")],
)


def test_from_runtime_sets_fields() -> None:
    runtime = MagicMock()
    reader = PoolReader.from_runtime(runtime, schema=_SCHEMA)
    assert reader.pool_name == _SCHEMA.name
    assert reader.schema == _SCHEMA


def test_from_runtime_does_not_own_runtime() -> None:
    runtime = MagicMock()
    reader = PoolReader.from_runtime(runtime, schema=_SCHEMA)
    reader.close()
    runtime.close.assert_not_called()


def test_close_is_idempotent() -> None:
    runtime = MagicMock()
    reader = PoolReader.from_runtime(runtime, schema=_SCHEMA)
    reader.close()
    reader.close()


def test_context_manager_calls_close() -> None:
    runtime = MagicMock()
    with PoolReader.from_runtime(runtime, schema=_SCHEMA) as reader:
        assert reader.pool_name == _SCHEMA.name


@patch("dr_llm.pool.reader.load_schema", return_value=None)
def test_open_raises_when_pool_not_in_catalog(mock_load: MagicMock) -> None:
    runtime = MagicMock()
    with pytest.raises(PoolNotFoundError):
        PoolReader.open("nonexistent_pool", runtime=runtime)
