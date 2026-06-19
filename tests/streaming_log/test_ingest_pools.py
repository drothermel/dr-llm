from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import cast

import pytest

import dr_llm.streaming_log.ingest_pools as ingest_pools_module
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log import StreamingEventLog
from dr_llm.streaming_log.events import StreamingLogEventType, idempotency_key
from dr_llm.streaming_log.ingest_pools import (
    PoolImportResult,
    PoolSnapshot,
    PoolSnapshotSource,
    ingest_pool,
    ingest_pools,
    record_pool_import,
)
from tests.streaming_log.helpers import (
    SpyStreamingEventLog,
    event_types,
    published_call,
)


class FailingSamples:
    def __iter__(self) -> Iterator[PoolSample]:
        return self

    def __next__(self) -> PoolSample:
        raise RuntimeError("source unavailable")


def _sample() -> PoolSample:
    return PoolSample(
        sample_id="sample-1",
        key_values={"dim": "a"},
        sample_idx=3,
        run_id="run-1",
        request={"prompt": "hello"},
        response={"text": "world"},
        finish_reason="stop",
        attempt_count=2,
        metadata={"m": 1},
    )


def _snapshot() -> PoolSnapshot:
    return PoolSnapshot(
        pool_name="pool-1",
        source_id="source-1",
        schema_payload={"name": "pool-1"},
    )


def _record_sample_import() -> tuple[SpyStreamingEventLog, PoolImportResult]:
    event_log = SpyStreamingEventLog()
    result = asyncio.run(
        record_pool_import(
            event_log=cast(StreamingEventLog, event_log),
            snapshot=_snapshot(),
            samples=[_sample()],
        )
    )
    return event_log, result


def test_record_pool_import_returns_result_event_ids() -> None:
    event_log, result = _record_sample_import()

    assert result.pool_name == "pool-1"
    assert result.imported_count == 1
    assert result.event_ids == [event.event_id for event in event_log.events]


def test_record_pool_import_emits_lifecycle_in_order() -> None:
    event_log, _ = _record_sample_import()

    assert event_types(event_log.published) == [
        StreamingLogEventType.pool_import_started,
        StreamingLogEventType.pool_sample_imported,
        StreamingLogEventType.pool_import_completed,
    ]


def test_record_pool_import_start_call_identifies_source() -> None:
    event_log, _ = _record_sample_import()

    started = published_call(
        event_log.published, StreamingLogEventType.pool_import_started
    )

    assert started.payload_roles == ["pool_schema"]
    assert started.idempotency_key == idempotency_key(
        "source-1", "pool-1", "pool_import_started"
    )
    assert started.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
    }
    assert started.context is not None
    assert started.context.source == "source-1"


def test_record_pool_import_sample_call_records_pool_row_identity() -> None:
    event_log, _ = _record_sample_import()

    imported = published_call(
        event_log.published, StreamingLogEventType.pool_sample_imported
    )

    assert imported.payload_roles == [
        "pool_schema",
        "request_json",
        "metadata_json",
        "response_json",
    ]
    assert imported.payload["pool_name"] == "pool-1"
    assert imported.payload["source_id"] == "source-1"
    assert imported.payload["sample_id"] == "sample-1"
    assert imported.payload["sample_idx"] == 3
    assert imported.payload["run_id"] == "run-1"
    assert imported.payload["key_values"] == {"dim": "a"}
    assert imported.payload["finish_reason"] == "stop"
    assert imported.payload["attempt_count"] == 2
    assert imported.payload["completion_state"] == "complete"
    assert imported.payload["reconstructed"] is True
    assert isinstance(imported.payload["row_state_hash"], str)
    assert imported.metadata == {"reconstructed": True}
    assert imported.context is not None
    assert imported.context.run_id == "run-1"
    assert imported.context.source == "source-1"


def test_record_pool_import_completed_call_records_count() -> None:
    event_log, _ = _record_sample_import()

    completed = published_call(
        event_log.published, StreamingLogEventType.pool_import_completed
    )

    assert completed.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
        "imported_count": 1,
        "reconstructed": True,
    }


def test_record_pool_import_records_source_iteration_failure() -> None:
    event_log = SpyStreamingEventLog()

    with pytest.raises(RuntimeError, match="source unavailable"):
        asyncio.run(
            record_pool_import(
                event_log=cast(StreamingEventLog, event_log),
                snapshot=_snapshot(),
                samples=FailingSamples(),
            )
        )

    assert event_types(event_log.published) == [
        StreamingLogEventType.pool_import_started,
        StreamingLogEventType.pool_import_failed,
    ]
    failed = published_call(
        event_log.published, StreamingLogEventType.pool_import_failed
    )
    assert failed.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
        "error_type": "RuntimeError",
        "message": "source unavailable",
    }


def test_record_pool_import_preserves_source_error_when_failure_event_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event_log = SpyStreamingEventLog(
        fail_on=StreamingLogEventType.pool_import_failed
    )

    with (
        caplog.at_level(
            logging.ERROR, logger="dr_llm.streaming_log.ingest_pools"
        ),
        pytest.raises(RuntimeError, match="source unavailable") as raised,
    ):
        asyncio.run(
            record_pool_import(
                event_log=cast(StreamingEventLog, event_log),
                snapshot=_snapshot(),
                samples=FailingSamples(),
            )
        )

    assert event_types(event_log.published) == [
        StreamingLogEventType.pool_import_started,
    ]
    notes = getattr(raised.value, "__notes__", [])
    assert notes == [
        "Publishing pool_import_failed event also failed: "
        "RuntimeError: failed to publish pool_import_failed"
    ]
    records = [
        record
        for record in caplog.records
        if record.name == "dr_llm.streaming_log.ingest_pools"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert "Failed to publish pool_import_failed event" in record.message
    assert "pool-1" in record.message
    assert "source-1" in record.message


def test_record_pool_import_does_not_emit_failed_for_publish_failure() -> None:
    event_log = SpyStreamingEventLog(
        fail_on=StreamingLogEventType.pool_sample_imported
    )

    with pytest.raises(
        RuntimeError, match="failed to publish pool_sample_imported"
    ):
        asyncio.run(
            record_pool_import(
                event_log=cast(StreamingEventLog, event_log),
                snapshot=_snapshot(),
                samples=[_sample()],
            )
        )

    assert event_types(event_log.published) == [
        StreamingLogEventType.pool_import_started,
    ]


def test_pool_snapshot_source_applies_sample_limit(monkeypatch) -> None:
    samples = [
        PoolSample(sample_id="sample-1"),
        PoolSample(sample_id="sample-2"),
        PoolSample(sample_id="sample-3"),
    ]
    runtime = FakeRuntime()
    reader = FakeReader(samples=samples)

    monkeypatch.setattr(
        ingest_pools_module,
        "_db_runtime",
        lambda *, dsn, application_name: runtime,
    )
    monkeypatch.setattr(
        ingest_pools_module.PoolReader,
        "open",
        lambda pool_name, *, runtime: reader,
    )

    with PoolSnapshotSource(
        dsn="postgresql://unused",
        pool_name="pool-1",
        source_id="source-1",
        sample_limit=2,
    ) as source:
        assert source.snapshot == PoolSnapshot(
            pool_name="pool-1",
            source_id="source-1",
            schema_payload={"name": "pool-1"},
        )
        assert [sample.sample_id for sample in source.samples()] == [
            "sample-1",
            "sample-2",
        ]

    assert runtime.closed


def test_ingest_pool_rejects_non_positive_sample_limit() -> None:
    with pytest.raises(ValueError, match="sample_limit must be at least 1"):
        asyncio.run(
            ingest_pool(
                event_log=cast(StreamingEventLog, None),
                dsn="postgresql://unused",
                pool_name="unused",
                sample_limit=0,
            )
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeSchema:
    def model_dump(
        self, *, mode: str, exclude_computed_fields: bool
    ) -> dict[str, str]:
        assert mode == "json"
        assert exclude_computed_fields
        return {"name": "pool-1"}


class FakeReader:
    def __init__(self, samples: list[PoolSample]) -> None:
        self.schema = FakeSchema()
        self._samples = samples

    def samples(self) -> Iterator[PoolSample]:
        yield from self._samples


def test_ingest_pools_rejects_non_positive_sample_limit() -> None:
    with pytest.raises(ValueError, match="sample_limit must be at least 1"):
        asyncio.run(
            ingest_pools(
                event_log=cast(StreamingEventLog, None),
                dsn="postgresql://unused",
                sample_limit=0,
            )
        )
