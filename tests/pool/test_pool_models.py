"""Unit tests for pool models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from dr_llm.pool.db import SampleColumn
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.results import InsertResult

_TEST_SCHEMA = PoolSchema(
    name="modeltest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)


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


def test_pool_sample_accepts_db_json_aliases() -> None:
    db_fields: Any = {
        "request_json": {"messages": []},
        "response_json": {"choices": []},
        "metadata_json": {"source": "test"},
    }
    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        **db_fields,
    )

    assert sample.request == {"messages": []}
    assert sample.response == {"choices": []}
    assert sample.metadata == {"source": "test"}


def test_pool_sample_accepts_python_field_names() -> None:
    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        request={"messages": []},
        response={"choices": []},
        metadata={"source": "test"},
    )

    assert sample.request == {"messages": []}
    assert sample.response == {"choices": []}
    assert sample.metadata == {"source": "test"}


def test_pool_sample_to_db_insert_row_splats_key_values() -> None:
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

    row = sample.to_db_insert_row()

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


def test_pool_sample_to_db_insert_row_json_serializes_nested_values() -> None:
    class RichPayload(BaseModel):
        when: datetime

    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        request={"rich": RichPayload(when=datetime(2024, 1, 2, tzinfo=UTC))},
        response={"when": datetime(2024, 1, 3, tzinfo=UTC)},
        metadata={"created_at": datetime(2024, 1, 4, tzinfo=UTC)},
    )

    row = sample.to_db_insert_row()

    assert row[SampleColumn.REQUEST_JSON] == {"rich": {"when": "2024-01-02T00:00:00Z"}}
    assert row[SampleColumn.RESPONSE_JSON] == {"when": "2024-01-03T00:00:00Z"}
    assert row[SampleColumn.METADATA_JSON] == {"created_at": "2024-01-04T00:00:00Z"}


def test_pool_sample_from_db_row_parses_dynamic_columns_and_json() -> None:
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    sample = PoolSample.from_db_row(
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


def test_pool_sample_from_db_row_round_trip_insert_row() -> None:
    sample = PoolSample.from_db_row(
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

    assert sample.to_db_insert_row() == {
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
