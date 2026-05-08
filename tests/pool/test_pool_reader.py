"""Unit tests for PoolReader pure-Python pieces."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.reader import (
    PoolProgress,
    PoolReader,
    PoolTableType,
    _validate_pool_name,
)

_TEST_SCHEMA = PoolSchema(
    name="reader_unit",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)


def test_pool_progress_in_flight_delegates_to_pending_counts() -> None:
    progress = PoolProgress(
        samples_total=10,
        pending_counts=PendingStatusCounts(pending=3, leased=2, failed=1),
    )
    assert progress.in_flight == 5
    assert progress.is_complete is False


def test_pool_progress_is_complete_when_no_in_flight() -> None:
    progress = PoolProgress(
        samples_total=10,
        pending_counts=PendingStatusCounts(pending=0, leased=0, failed=2),
    )
    assert progress.in_flight == 0
    assert progress.is_complete is True


def test_pool_progress_computed_fields_appear_in_model_dump() -> None:
    """Derived properties must be exposed via @computed_field so they
    survive model_dump / JSON serialization."""
    progress = PoolProgress(
        samples_total=42,
        pending_counts=PendingStatusCounts(pending=1, leased=2, failed=0),
    )
    dumped = progress.model_dump()
    assert dumped["samples_total"] == 42
    assert dumped["in_flight"] == 3
    assert dumped["is_complete"] is False
    # pending_counts is a nested model; its own derived fields are also
    # subject to whether they were declared with @computed_field. The
    # existing PendingStatusCounts uses plain @property so its in_flight
    # is NOT in the dump — that's out of scope for this PR.
    assert "pending_counts" in dumped


def test_pool_progress_is_frozen() -> None:
    progress = PoolProgress(
        samples_total=1,
        pending_counts=PendingStatusCounts(),
    )
    with pytest.raises(Exception, match="frozen"):
        progress.samples_total = 99


def test_pool_table_type_values_are_stable() -> None:
    assert PoolTableType.SAMPLES.value == "samples"
    assert PoolTableType.PENDING.value == "pending"
    assert PoolTableType.CLAIMS.value == "claims"
    assert PoolTableType.METADATA.value == "metadata"
    assert PoolTableType.CALL_STATS.value == "call_stats"


def test_load_table_df_returns_empty_frame_with_expected_columns() -> None:
    runtime = MagicMock()
    connection = runtime.connect.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value.all.return_value = []
    reader = PoolReader.from_runtime(runtime, schema=_TEST_SCHEMA)

    frame = reader.load_table_df(PoolTableType.CALL_STATS)

    assert frame.empty
    assert list(frame.columns) == [
        "sample_id",
        "latency_ms",
        "total_cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "attempt_count",
        "finish_reason",
        "created_at",
    ]


def test_dataframe_convenience_methods_delegate_to_enum_loader() -> None:
    reader = PoolReader.__new__(PoolReader)
    reader.load_table_df = Mock(return_value="frame")

    assert reader.samples_df() == "frame"
    reader.load_table_df.assert_called_with(PoolTableType.SAMPLES)
    assert reader.pending_df() == "frame"
    reader.load_table_df.assert_called_with(PoolTableType.PENDING)
    assert reader.claims_df() == "frame"
    reader.load_table_df.assert_called_with(PoolTableType.CLAIMS)
    assert reader.metadata_df() == "frame"
    reader.load_table_df.assert_called_with(PoolTableType.METADATA)
    assert reader.call_stats_df() == "frame"
    reader.load_table_df.assert_called_with(PoolTableType.CALL_STATS)


@pytest.mark.parametrize(
    "name",
    ["a", "abc", "a1", "snake_case", "with_123_digits"],
)
def test_validate_pool_name_accepts_valid(name: str) -> None:
    _validate_pool_name(name)  # should not raise


@pytest.mark.parametrize(
    "name",
    [
        "",
        "1starts_with_digit",
        "_underscore_start",
        "Capital",
        "has-dash",
        "has.dot",
        "has space",
        "has;semicolon",  # SQL injection style
        'has"quote',
    ],
)
def test_validate_pool_name_rejects_invalid(name: str) -> None:
    with pytest.raises(ValueError, match="pool_name must be"):
        _validate_pool_name(name)


def test_samples_warns_and_ignores_legacy_status_argument() -> None:
    reader = PoolReader.__new__(PoolReader)
    store = Mock()
    expected = iter(["sample"])
    store.iter_samples.return_value = expected
    reader._store = store

    with pytest.deprecated_call(match="status argument is ignored and will be removed"):
        result = reader.samples(status="active")

    assert result is expected
    store.iter_samples.assert_called_once_with(key_filter=None)


def test_samples_list_warns_and_ignores_legacy_status_argument() -> None:
    reader = PoolReader.__new__(PoolReader)
    store = Mock()
    store.bulk_load.return_value = ["sample"]
    reader._store = store

    with pytest.deprecated_call(match="status argument is ignored and will be removed"):
        result = reader.samples_list(status="active")

    assert result == ["sample"]
    store.bulk_load.assert_called_once_with(key_filter=None)
