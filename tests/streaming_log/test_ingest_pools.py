from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

import dr_llm.streaming_log.ingest_pools as ingest_pools_module
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log import StreamingLogClient
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    build_event,
    idempotency_key,
    stable_hash,
)
from dr_llm.streaming_log.ingest_pools import (
    PoolSnapshot,
    PoolSnapshotSource,
    _sample_snapshot_payload,
    ingest_pool,
    ingest_pools,
    record_pool_import,
)
from dr_llm.streaming_log.payloads import PreparedPayload


class PublishedEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: EventEnvelope
    payload_roles: list[str] = Field(default_factory=list)


class FakeClient:
    def __init__(self, fail_on: StreamingLogEventType | None = None) -> None:
        self.fail_on = fail_on
        self.published: list[PublishedEvent] = []

    async def publish_event_with_payloads(
        self,
        event_type: StreamingLogEventType,
        *,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        payloads: list[PreparedPayload] | None = None,
        context: EventContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        if event_type == self.fail_on:
            raise RuntimeError(f"failed to publish {event_type}")
        payloads = payloads or []
        event = build_event(
            event_type,
            producer=ProducerInfo(name="test"),
            idempotency_key=idempotency_key,
            payload=payload,
            payload_refs=[item.ref() for item in payloads],
            context=context,
            metadata=metadata,
        )
        self.published.append(
            PublishedEvent(
                event=event,
                payload_roles=[item.role for item in payloads],
            )
        )
        return event

    @property
    def events(self) -> list[EventEnvelope]:
        return [record.event for record in self.published]


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


def _event_types(client: FakeClient) -> list[StreamingLogEventType]:
    return [event.event_type for event in client.events]


def _published(
    client: FakeClient, event_type: StreamingLogEventType
) -> PublishedEvent:
    for record in client.published:
        if record.event.event_type == event_type:
            return record
    raise AssertionError(f"missing event {event_type}")


def test_sample_snapshot_payload_preserves_pool_row_state() -> None:
    sample = _sample()

    payload = _sample_snapshot_payload(sample)

    assert payload["sample_id"] == "sample-1"
    assert payload["key_values"] == {"dim": "a"}
    assert payload["sample_idx"] == 3
    assert payload["run_id"] == "run-1"
    assert payload["request"] == {"prompt": "hello"}
    assert payload["response"] == {"text": "world"}
    assert payload["finish_reason"] == "stop"
    assert payload["attempt_count"] == 2
    assert payload["metadata"] == {"m": 1}


def test_record_pool_import_emits_lifecycle_payloads() -> None:
    client = FakeClient()
    sample = _sample()
    snapshot = _snapshot()

    result = asyncio.run(
        record_pool_import(
            client=cast(StreamingLogClient, client),
            snapshot=snapshot,
            samples=[sample],
        )
    )

    assert result.pool_name == "pool-1"
    assert result.imported_count == 1
    assert result.event_ids == [event.event_id for event in client.events]
    assert _event_types(client) == [
        StreamingLogEventType.pool_import_started,
        StreamingLogEventType.pool_sample_imported,
        StreamingLogEventType.pool_import_completed,
    ]
    started = _published(client, StreamingLogEventType.pool_import_started)
    assert started.payload_roles == ["pool_schema"]
    assert started.event.idempotency_key == idempotency_key(
        "source-1", "pool-1", "pool_import_started"
    )
    assert started.event.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
    }
    assert started.event.source == "source-1"

    imported = _published(client, StreamingLogEventType.pool_sample_imported)
    state_hash = stable_hash(_sample_snapshot_payload(sample))
    assert imported.payload_roles == [
        "pool_schema",
        "request_json",
        "metadata_json",
        "response_json",
    ]
    assert imported.event.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
        "sample_id": "sample-1",
        "sample_idx": 3,
        "run_id": "run-1",
        "key_values": {"dim": "a"},
        "finish_reason": "stop",
        "attempt_count": 2,
        "created_at": None,
        "completion_state": "complete",
        "reconstructed": True,
        "row_state_hash": state_hash,
    }
    assert imported.event.metadata == {"reconstructed": True}
    assert imported.event.run_id == "run-1"
    assert imported.event.source == "source-1"

    completed = _published(client, StreamingLogEventType.pool_import_completed)
    assert completed.event.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
        "imported_count": 1,
        "reconstructed": True,
    }


def test_record_pool_import_records_source_iteration_failure() -> None:
    client = FakeClient()

    with pytest.raises(RuntimeError, match="source unavailable"):
        asyncio.run(
            record_pool_import(
                client=cast(StreamingLogClient, client),
                snapshot=_snapshot(),
                samples=FailingSamples(),
            )
        )

    assert _event_types(client) == [
        StreamingLogEventType.pool_import_started,
        StreamingLogEventType.pool_import_failed,
    ]
    failed = _published(client, StreamingLogEventType.pool_import_failed)
    assert failed.event.payload == {
        "pool_name": "pool-1",
        "source_id": "source-1",
        "error_type": "RuntimeError",
        "message": "source unavailable",
    }


def test_record_pool_import_preserves_source_error_when_failure_event_fails() -> (
    None
):
    client = FakeClient(fail_on=StreamingLogEventType.pool_import_failed)

    with pytest.raises(RuntimeError, match="source unavailable"):
        asyncio.run(
            record_pool_import(
                client=cast(StreamingLogClient, client),
                snapshot=_snapshot(),
                samples=FailingSamples(),
            )
        )

    assert _event_types(client) == [
        StreamingLogEventType.pool_import_started,
    ]


def test_record_pool_import_does_not_emit_failed_for_publish_failure() -> None:
    client = FakeClient(fail_on=StreamingLogEventType.pool_sample_imported)

    with pytest.raises(
        RuntimeError, match="failed to publish pool_sample_imported"
    ):
        asyncio.run(
            record_pool_import(
                client=cast(StreamingLogClient, client),
                snapshot=_snapshot(),
                samples=[_sample()],
            )
        )

    assert _event_types(client) == [
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
                client=cast(StreamingLogClient, None),
                dsn="postgresql://unused",
                pool_name="unused",
                sample_limit=0,
            )
        )


class FakeRuntime:
    closed = False

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
                client=cast(StreamingLogClient, None),
                dsn="postgresql://unused",
                sample_limit=0,
            )
        )
