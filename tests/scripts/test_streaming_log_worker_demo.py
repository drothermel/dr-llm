from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
import asyncio
import pytest

from typer.testing import CliRunner

from dr_llm.llm import ProviderName
from dr_llm.streaming_log.events import (
    AttemptFailedPayload,
    AttemptStartedPayload,
    AttemptSucceededPayload,
    EventEnvelope,
    ProducerInfo,
    ProducerLifecyclePayload,
    ProviderRequestPreparedPayload,
    ProviderResponseReceivedPayload,
    StreamingLogEventType,
    WorkCompletedPayload,
    WorkSubmittedPayload,
)
from dr_llm.streaming_log.payloads import PayloadRef, prepare_json_payload


runner = CliRunner()


def _load_worker_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_worker", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_demo_command_forwards_provider_options(monkeypatch) -> None:
    worker_demo = _load_worker_demo()
    calls: list[str] = []

    async def fake_run_worker_demo(options: Any) -> None:
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.provider == ProviderName.ANTHROPIC
        assert options.prompt == "hello"
        assert options.max_retries == 2
        assert options.model == "claude-test"
        calls.append("run")

    monkeypatch.setattr(
        worker_demo,
        "_run_worker_demo",
        fake_run_worker_demo,
    )

    result = runner.invoke(
        worker_demo.app,
        [
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--prompt",
            "hello",
            "--max-retries",
            "2",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]


def test_worker_lifecycle_accepts_successful_selected_work() -> None:
    worker_demo = _load_worker_demo()

    worker_demo._verify_worker_lifecycle(
        _successful_events(), work_id="work-1", max_retries=0
    )


def test_worker_lifecycle_rejects_clean_provider_failure() -> None:
    worker_demo = _load_worker_demo()

    with pytest.raises(
        worker_demo.ProviderWorkFailedError, match="BillingError"
    ):
        worker_demo._verify_worker_lifecycle(
            _failed_events(), work_id="work-1", max_retries=0
        )


def test_worker_lifecycle_rejects_non_successful_completion() -> None:
    worker_demo = _load_worker_demo()
    events = _successful_events()
    events[-2] = _event(
        StreamingLogEventType.work_completed,
        WorkCompletedPayload(status="failed", attempt=1),
    )

    with pytest.raises(
        worker_demo.ProviderWorkFailedError,
        match="work_completed status is 'failed'",
    ):
        worker_demo._verify_worker_lifecycle(
            events, work_id="work-1", max_retries=0
        )


def test_worker_response_payload_requires_non_empty_text() -> None:
    worker_demo = _load_worker_demo()
    payload = prepare_json_payload("response_json", {"text": "OK."})
    events = _successful_events(response_ref=payload.ref())

    response = asyncio.run(
        worker_demo._verify_response_payload(
            _FakePayloadStore({payload.object_key: payload.data}),
            events,
            work_id="work-1",
        )
    )

    assert response == {"text_preview": "OK."}


def test_worker_response_payload_rejects_empty_text() -> None:
    worker_demo = _load_worker_demo()
    payload = prepare_json_payload("response_json", {"text": ""})
    events = _successful_events(response_ref=payload.ref())

    with pytest.raises(RuntimeError, match="non-empty text"):
        asyncio.run(
            worker_demo._verify_response_payload(
                _FakePayloadStore({payload.object_key: payload.data}),
                events,
                work_id="work-1",
            )
        )


class _FakePayloadStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    async def read_payload_ref(self, payload_ref: PayloadRef) -> bytes:
        return self.payloads[payload_ref.object_key]


def _successful_events(
    response_ref: PayloadRef | None = None,
) -> list[EventEnvelope]:
    return [
        _producer_event(StreamingLogEventType.producer_started),
        _event(
            StreamingLogEventType.work_submitted,
            WorkSubmittedPayload(work_id="work-1", max_retries=0),
        ),
        _event(
            StreamingLogEventType.attempt_started,
            AttemptStartedPayload(worker_id="worker-1", attempt=1),
        ),
        _event(
            StreamingLogEventType.provider_request_prepared,
            ProviderRequestPreparedPayload(
                provider="openai", model="gpt-test", mode="api"
            ),
        ),
        _event(
            StreamingLogEventType.provider_response_received,
            ProviderResponseReceivedPayload(
                provider="openai", model="gpt-test", mode="api"
            ),
            payload_refs=[response_ref] if response_ref is not None else [],
        ),
        _event(
            StreamingLogEventType.attempt_succeeded,
            AttemptSucceededPayload(attempt=1),
        ),
        _event(
            StreamingLogEventType.work_completed,
            WorkCompletedPayload(status="succeeded", attempt=1),
        ),
        _producer_event(StreamingLogEventType.producer_stopped),
    ]


def _failed_events() -> list[EventEnvelope]:
    return [
        _producer_event(StreamingLogEventType.producer_started),
        _event(
            StreamingLogEventType.work_submitted,
            WorkSubmittedPayload(work_id="work-1", max_retries=0),
        ),
        _event(
            StreamingLogEventType.attempt_started,
            AttemptStartedPayload(worker_id="worker-1", attempt=1),
        ),
        _event(
            StreamingLogEventType.provider_request_prepared,
            ProviderRequestPreparedPayload(
                provider="anthropic", model="claude-test", mode="api"
            ),
        ),
        _event(
            StreamingLogEventType.attempt_failed,
            AttemptFailedPayload(
                error_type="BillingError",
                message="billing disabled",
                attempt=1,
            ),
        ),
        _event(
            StreamingLogEventType.work_completed,
            WorkCompletedPayload(
                status="failed",
                attempt=1,
                error_type="BillingError",
                message="billing disabled",
            ),
        ),
        _producer_event(StreamingLogEventType.producer_stopped),
    ]


def _event(
    event_type: StreamingLogEventType,
    payload: Any,
    *,
    payload_refs: list[PayloadRef] | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"{event_type}-idem",
        payload=payload,
        work_id="work-1",
        attempt_id="attempt-1",
        payload_refs=payload_refs or [],
    )


def _producer_event(event_type: StreamingLogEventType) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"{event_type}-idem",
        payload=ProducerLifecyclePayload(worker_id="worker-1"),
    )
