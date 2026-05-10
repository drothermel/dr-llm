"""Unit tests for pool models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from dr_llm.pool.db import SampleColumn
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.tables import SamplesTableDef
from dr_llm.pool.insert_result import InsertResult
from dr_llm.pool.pool_progress import PoolProgress
from dr_llm.pool.pool_sample import PoolSample

_TEST_SCHEMA = PoolSchema(
    name="modeltest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_SAMPLES_DEF = SamplesTableDef()


def test_pool_sample_defaults() -> None:
    sample = PoolSample(key_values={"x": "a"})
    assert sample.sample_id
    assert sample.sample_idx is None
    assert sample.run_id is None
    assert sample.request == {}
    assert sample.response is None
    assert sample.finish_reason is None
    assert sample.attempt_count == 0
    assert sample.metadata == {}
    assert sample.is_complete is False


def test_pool_sample_is_complete_for_response() -> None:
    sample = PoolSample(key_values={"x": "a"}, response={"id": "response-1"})
    assert sample.is_complete is True


def test_pool_sample_field_names() -> None:
    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        request={"messages": []},
        response={"choices": []},
        metadata={"source": "test"},
    )

    assert sample.request == {"messages": []}
    assert sample.response == {"choices": []}
    assert sample.metadata == {"source": "test"}


def test_sample_to_row_splats_key_values() -> None:
    sample = PoolSample(
        sample_id="sample-1",
        sample_idx=7,
        key_values={"dim_b": 3, "dim_a": "alpha"},
        run_id="run-1",
        request={"messages": [{"role": "user", "content": "Hi"}]},
        response={"finish_reason": "stop"},
        finish_reason="stop",
        attempt_count=2,
        metadata={"source": "test"},
    )

    row = _SAMPLES_DEF.sample_to_row(sample)

    assert set(row.keys()) == {
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
    }
    assert row[SampleColumn.SAMPLE_ID] == "sample-1"
    assert row[SampleColumn.SAMPLE_IDX] == 7
    assert row["dim_a"] == "alpha"
    assert row["dim_b"] == 3
    assert row[SampleColumn.RUN_ID] == "run-1"
    assert row[SampleColumn.REQUEST_JSON] == {
        "messages": [{"role": "user", "content": "Hi"}]
    }
    assert row[SampleColumn.RESPONSE_JSON] == {"finish_reason": "stop"}
    assert row[SampleColumn.FINISH_REASON] == "stop"
    assert row[SampleColumn.ATTEMPT_COUNT] == 2
    assert row[SampleColumn.METADATA_JSON] == {"source": "test"}


def test_sample_to_row_json_serializes_nested_values() -> None:
    class RichPayload(BaseModel):
        when: datetime

    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        request={"rich": RichPayload(when=datetime(2024, 1, 2, tzinfo=UTC))},
        response={"when": datetime(2024, 1, 3, tzinfo=UTC)},
        metadata={"created_at": datetime(2024, 1, 4, tzinfo=UTC)},
    )

    row = _SAMPLES_DEF.sample_to_row(sample)

    assert row[SampleColumn.REQUEST_JSON] == {"rich": {"when": "2024-01-02T00:00:00Z"}}
    assert row[SampleColumn.RESPONSE_JSON] == {"when": "2024-01-03T00:00:00Z"}
    assert row[SampleColumn.METADATA_JSON] == {"created_at": "2024-01-04T00:00:00Z"}


def test_sample_from_row_parses_dynamic_columns_and_json() -> None:
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    sample = _SAMPLES_DEF.sample_from_row(
        _TEST_SCHEMA,
        {
            "sample_id": "sample-1",
            "dim_a": "alpha",
            "dim_b": 3,
            "sample_idx": 7,
            "run_id": "run-1",
            "request_json": {"messages": []},
            "response_json": {"finish_reason": "stop"},
            "finish_reason": "stop",
            "attempt_count": 2,
            "metadata_json": {"source": "test"},
            "created_at": created_at,
        },
    )

    assert sample.sample_id == "sample-1"
    assert sample.key_values == {"dim_a": "alpha", "dim_b": 3}
    assert sample.sample_idx == 7
    assert sample.run_id == "run-1"
    assert sample.request == {"messages": []}
    assert sample.response == {"finish_reason": "stop"}
    assert sample.finish_reason == "stop"
    assert sample.attempt_count == 2
    assert sample.metadata == {"source": "test"}
    assert sample.created_at == created_at
    assert sample.is_complete is True


def test_sample_row_round_trip() -> None:
    sample = _SAMPLES_DEF.sample_from_row(
        _TEST_SCHEMA,
        {
            "sample_id": "sample-1",
            "dim_a": "alpha",
            "dim_b": 3,
            "sample_idx": 7,
            "run_id": None,
            "request_json": {"messages": []},
            "response_json": None,
            "finish_reason": None,
            "attempt_count": 0,
            "metadata_json": {},
        },
    )

    assert _SAMPLES_DEF.sample_to_row(sample) == {
        SampleColumn.SAMPLE_ID: "sample-1",
        SampleColumn.SAMPLE_IDX: 7,
        SampleColumn.RUN_ID: None,
        SampleColumn.REQUEST_JSON: {"messages": []},
        SampleColumn.RESPONSE_JSON: None,
        SampleColumn.FINISH_REASON: None,
        SampleColumn.ATTEMPT_COUNT: 0,
        SampleColumn.METADATA_JSON: {},
        "dim_a": "alpha",
        "dim_b": 3,
    }
    assert sample.is_complete is False


def test_insert_result_defaults() -> None:
    result = InsertResult()
    assert result.inserted == 0
    assert result.skipped == 0
    assert result.failed == 0


def test_pool_progress_construction() -> None:
    p = PoolProgress(total=10, incomplete=4, leased=2, complete=6, error=1)
    assert p.total == 10
    assert p.incomplete == 4
    assert p.leased == 2
    assert p.complete == 6
    assert p.error == 1


@pytest.mark.parametrize(
    "field_name", ["total", "incomplete", "leased", "complete", "error"]
)
def test_pool_progress_rejects_negative_counts(field_name: str) -> None:
    values = {"total": 10, "incomplete": 4, "leased": 2, "complete": 6, "error": 1}
    values[field_name] = -1

    with pytest.raises(ValueError, match=rf"PoolProgress\.{field_name} must be >= 0"):
        PoolProgress(**values)
