from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dr_llm.artifact_projection import (
    ArtifactIndexSummary,
    ArtifactProjectionConfig,
    ArtifactStore,
    PayloadArtifactSource,
    ProjectionError,
    ProjectionErrorKind,
)
from dr_llm.artifact_projection.index import ArtifactIndex
from dr_llm.artifact_projection.projector import (
    ArtifactEventDelivery,
    ArtifactProjector,
    stream_sequence_for_message,
)
from dr_llm.artifact_projection.storage import ArtifactReader
from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    ProviderResponseReceivedPayload,
    StreamingLogEventType,
)
from dr_llm.streaming_log.errors import PayloadNotFoundError, PayloadReadError
from dr_llm.streaming_log.payloads import PayloadRef, prepare_json_payload


class FakePayloadReader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    async def read_payload_ref(self, payload_ref: PayloadRef) -> bytes:
        try:
            return self.payloads[payload_ref.object_key]
        except KeyError as exc:
            raise PayloadNotFoundError("missing test payload") from exc


class FailingPayloadReader:
    async def read_payload_ref(self, payload_ref: PayloadRef) -> bytes:
        raise PayloadReadError("test read failure")


class FakeMessage:
    def __init__(self, event: EventEnvelope) -> None:
        self.data = event.json_bytes()
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


def test_projector_writes_artifact_before_ack(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FakePayloadReader({payload.object_key: payload.data}),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=7, message=message
            )
        )
    )

    assert result.projected_count == 1
    assert message.acked
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.artifact_count == 1
    assert summary.checkpoint is not None
    reference = store.index.list_references()[0]
    source = PayloadArtifactSource.from_event_ref(
        event=event, payload_ref=payload.ref()
    )
    assert reference.source_ref == source.source_ref
    assert reference.event_context == source.event_context
    assert ArtifactReader(config).read_json(reference) == {"ok": True}


def test_projector_skips_duplicate_ref_in_same_event(tmp_path: Path) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref(), payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FakePayloadReader({payload.object_key: payload.data}),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=7, message=message
            )
        )
    )

    assert result.projected_count == 1
    assert result.skipped_count == 1
    assert message.acked
    summary = store.index.summary(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
    )
    assert summary.artifact_count == 1
    assert summary.open_artifact_count == 0


def test_projector_records_missing_payload_error_and_acks(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FakePayloadReader({}),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=8, message=message
            )
        )
    )

    assert result.error_count == 1
    assert message.acked
    error = _single_projection_error(config)
    summary = _summary(config)
    assert summary.error_count == 1
    assert summary.artifact_count == 0
    assert error.source_ref.event_id == event.event_id
    assert error.source_ref.idempotency_key == "idem-1"
    assert error.source_ref.payload_role == "response_json"
    assert error.event_context.run_id == "run-1"
    assert error.event_context.metadata == {"purpose": "projection-test"}
    assert error.error_kind is ProjectionErrorKind.missing_payload


def test_projector_records_payload_read_error_and_acks(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FailingPayloadReader(),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=8, message=message
            )
        )
    )

    assert result.error_count == 1
    assert message.acked
    assert _single_projection_error(config).error_kind is (
        ProjectionErrorKind.storage_error
    )


def test_projector_records_payload_size_mismatch_error_and_acks(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FakePayloadReader({payload.object_key: b"too-small"}),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=9, message=message
            )
        )
    )

    assert result.projected_count == 0
    assert result.error_count == 1
    assert message.acked
    assert _summary(config).artifact_count == 0
    assert _single_projection_error(config).error_kind is (
        ProjectionErrorKind.source_size_mismatch
    )


def test_projector_records_payload_hash_mismatch_error_and_acks(
    tmp_path: Path,
) -> None:
    config = ArtifactProjectionConfig(artifact_root=tmp_path)
    store = ArtifactStore(config=config)
    store.initialize()
    payload = prepare_json_payload("response_json", {"ok": True})
    event = _event(payload.ref())
    message = FakeMessage(event)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=FakePayloadReader({payload.object_key: b'{"ok":null}'}),
    )

    result = _run(
        projector.process_delivery(
            ArtifactEventDelivery(
                event=event, stream_sequence=10, message=message
            )
        )
    )

    assert result.projected_count == 0
    assert result.error_count == 1
    assert message.acked
    assert _summary(config).artifact_count == 0
    assert _single_projection_error(config).error_kind is (
        ProjectionErrorKind.source_hash_mismatch
    )


def test_stream_sequence_for_message_reads_metadata() -> None:
    message = SimpleNamespace(
        metadata=SimpleNamespace(sequence=SimpleNamespace(stream=42))
    )

    assert stream_sequence_for_message(message) == 42


def test_stream_sequence_for_message_requires_stream_sequence() -> None:
    message = SimpleNamespace(metadata=SimpleNamespace(sequence=object()))

    with pytest.raises(ValueError, match="missing stream sequence"):
        stream_sequence_for_message(message)


def _event(*payload_refs: PayloadRef) -> EventEnvelope:
    return EventEnvelope(
        event_type=StreamingLogEventType.provider_response_received,
        producer=ProducerInfo(name="test"),
        idempotency_key="idem-1",
        payload=ProviderResponseReceivedPayload(
            provider="test",
            model="test-model",
            mode="api",
        ),
        payload_refs=list(payload_refs),
        run_id="run-1",
        source="test-suite",
        metadata={"purpose": "projection-test"},
    )


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _summary(config: ArtifactProjectionConfig) -> ArtifactIndexSummary:
    with ArtifactIndex(config.index_path) as index:
        return index.summary(
            projection_version=config.projection_version,
            durable_consumer=config.durable_consumer,
        )


def _single_projection_error(
    config: ArtifactProjectionConfig,
) -> ProjectionError:
    with ArtifactIndex(config.index_path) as index:
        rows = index.connection.execute(
            """
            SELECT error_json
            FROM projection_errors
            ORDER BY error_id
            """
        ).fetchall()
    assert len(rows) == 1
    return ProjectionError.model_validate_json(rows[0]["error_json"])
