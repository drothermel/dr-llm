from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    idempotency_key,
)
from dr_llm.streaming_log.payloads import (
    object_key_for_sha256,
    prepare_json_payload,
    prepare_text_payload,
    sha256_bytes,
)


def test_event_envelope_requires_strict_event_type() -> None:
    event = EventEnvelope(
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(name="test", version="1"),
        idempotency_key="same",
    )

    assert event.event_type is StreamingLogEventType.work_submitted
    assert event.occurred_at.tzinfo is not None


def test_event_envelope_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EventEnvelope(
            event_type=StreamingLogEventType.work_submitted,
            producer=ProducerInfo(),
            idempotency_key="same",
            occurred_at=datetime(2026, 1, 1),
        )


def test_idempotency_key_is_deterministic() -> None:
    first = idempotency_key("pool", 1, {"a": "b"})
    second = idempotency_key("pool", 1, {"a": "b"})
    different = idempotency_key("pool", 2, {"a": "b"})

    assert first == second
    assert first != different


def test_payload_hash_and_object_key_are_content_addressed() -> None:
    payload = prepare_text_payload("stdout", "hello")
    digest = sha256_bytes(b"hello")

    assert payload.sha256 == digest
    assert payload.object_key == f"sha256/{digest[:2]}/{digest}"
    assert payload.ref().model_dump(mode="json") == {
        "role": "stdout",
        "object_key": f"sha256/{digest[:2]}/{digest}",
        "sha256": digest,
        "size_bytes": 5,
        "content_type": "text/plain",
        "encoding": "utf-8",
        "compression": "none",
    }


def test_json_payload_serialization_is_stable() -> None:
    left = prepare_json_payload("request_json", {"b": 2, "a": 1})
    right = prepare_json_payload("request_json", {"a": 1, "b": 2})

    assert left.data == b'{"a":1,"b":2}'
    assert left.sha256 == right.sha256


def test_object_key_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="64 hex"):
        object_key_for_sha256("nope")


def test_timestamp_is_normalized_to_utc() -> None:
    event = EventEnvelope(
        event_type=StreamingLogEventType.producer_started,
        producer=ProducerInfo(),
        idempotency_key="same",
        occurred_at=datetime.now(timezone.utc),
    )

    assert event.occurred_at.tzinfo is timezone.utc
