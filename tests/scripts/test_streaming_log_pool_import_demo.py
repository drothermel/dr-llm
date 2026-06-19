from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

from dr_llm.streaming_log.events import (
    EventEnvelope,
    PoolImportCompletedPayload,
    PoolImportFailedPayload,
    PoolImportStartedPayload,
    PoolSampleImportedPayload,
    ProducerInfo,
    StreamingLogEventType,
)
from dr_llm.streaming_log.ingest_pools import PoolImportResult
from dr_llm.streaming_log.payloads import PayloadRef, prepare_json_payload


runner = CliRunner()


def _load_pool_import_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-pool-import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_pool_import", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pool_import_demo_command_forwards_options(monkeypatch) -> None:
    pool_import_demo = _load_pool_import_demo()
    calls: list[str] = []

    async def fake_run_import_demo(options: Any) -> None:
        assert options.dsn == "postgresql://localhost/demo"
        assert options.pool_name == "demo_pool"
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.source_id == "source"
        assert options.sample_limit == 3
        assert options.event_sample_limit == 2
        calls.append("run")

    monkeypatch.setattr(
        pool_import_demo,
        "_run_import_demo",
        fake_run_import_demo,
    )

    result = runner.invoke(
        pool_import_demo.app,
        [
            "--dsn",
            "postgresql://localhost/demo",
            "--pool-name",
            "demo_pool",
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--source-id",
            "source",
            "--sample-limit",
            "3",
            "--event-sample-limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]


def test_pool_import_events_accept_matching_import() -> None:
    pool_import_demo = _load_pool_import_demo()
    events = _successful_import_events()

    pool_import_demo._verify_import_events(
        expected_import_count=1,
        result=PoolImportResult(
            pool_name="pool-1",
            imported_count=1,
            event_ids=[event.event_id for event in events],
        ),
        events=events,
        pool_name="pool-1",
        source_id="source-1",
    )


def test_pool_import_events_reject_failed_import_event() -> None:
    pool_import_demo = _load_pool_import_demo()
    events = [
        *_successful_import_events(),
        _event(
            "event-4",
            StreamingLogEventType.pool_import_failed,
            PoolImportFailedPayload(
                pool_name="pool-1",
                source_id="source-1",
                error_type="RuntimeError",
                message="boom",
            ),
        ),
    ]

    with pytest.raises(RuntimeError, match="pool_import_failed"):
        pool_import_demo._verify_import_events(
            expected_import_count=1,
            result=PoolImportResult(
                pool_name="pool-1",
                imported_count=1,
                event_ids=[event.event_id for event in events],
            ),
            events=events,
            pool_name="pool-1",
            source_id="source-1",
        )


def test_pool_import_events_reject_completed_count_mismatch() -> None:
    pool_import_demo = _load_pool_import_demo()
    events = _successful_import_events(completed_count=2)

    with pytest.raises(RuntimeError, match="imported_count mismatch"):
        pool_import_demo._verify_import_events(
            expected_import_count=1,
            result=PoolImportResult(
                pool_name="pool-1",
                imported_count=1,
                event_ids=[event.event_id for event in events],
            ),
            events=events,
            pool_name="pool-1",
            source_id="source-1",
        )


def test_pool_import_events_reject_missing_sample_payload_role() -> None:
    pool_import_demo = _load_pool_import_demo()
    events = _successful_import_events(sample_refs=_sample_refs()[:2])

    with pytest.raises(RuntimeError, match="metadata_json"):
        pool_import_demo._verify_import_events(
            expected_import_count=1,
            result=PoolImportResult(
                pool_name="pool-1",
                imported_count=1,
                event_ids=[event.event_id for event in events],
            ),
            events=events,
            pool_name="pool-1",
            source_id="source-1",
        )


def _successful_import_events(
    *,
    completed_count: int = 1,
    sample_refs: list[PayloadRef] | None = None,
) -> list[EventEnvelope]:
    return [
        _event(
            "event-1",
            StreamingLogEventType.pool_import_started,
            PoolImportStartedPayload(pool_name="pool-1", source_id="source-1"),
        ),
        _event(
            "event-2",
            StreamingLogEventType.pool_sample_imported,
            PoolSampleImportedPayload(
                pool_name="pool-1",
                source_id="source-1",
                sample_id="sample-1",
                key_values={"case": "a"},
                attempt_count=0,
                completion_state="complete",
                reconstructed=True,
                row_state_hash="abc123",
            ),
            payload_refs=sample_refs or _sample_refs(),
        ),
        _event(
            "event-3",
            StreamingLogEventType.pool_import_completed,
            PoolImportCompletedPayload(
                pool_name="pool-1",
                source_id="source-1",
                imported_count=completed_count,
                reconstructed=True,
            ),
        ),
    ]


def _sample_refs() -> list[PayloadRef]:
    return [
        prepare_json_payload("pool_schema", {"name": "pool-1"}).ref(),
        prepare_json_payload("request_json", {"prompt": "hello"}).ref(),
        prepare_json_payload("metadata_json", {"meta": True}).ref(),
    ]


def _event(
    event_id: str,
    event_type: StreamingLogEventType,
    payload: Any,
    *,
    payload_refs: list[PayloadRef] | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"{event_type}-idem",
        payload=payload,
        payload_refs=payload_refs or [],
    )
