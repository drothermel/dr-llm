from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from typer.testing import CliRunner

from dr_llm.streaming_log.events import (
    AttemptFailedPayload,
    AttemptStartedPayload,
    AttemptSucceededPayload,
    EventEnvelope,
    ProducerInfo,
    ProviderRequestPreparedPayload,
    ProviderResponseReceivedPayload,
    StreamingLogEventType,
    WorkCompletedPayload,
    WorkSubmittedPayload,
)

runner = CliRunner()


def _load_metadata_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-metadata-projection-e2e.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_metadata_projection_e2e", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_projection_e2e_demo_help_smoke() -> None:
    metadata_demo = _load_metadata_demo()

    result = runner.invoke(metadata_demo.app, ["--help"])

    assert result.exit_code == 0


def test_metadata_projection_e2e_demo_forwards_options(
    tmp_path: Path, monkeypatch
) -> None:
    metadata_demo = _load_metadata_demo()
    calls: list[str] = []

    async def fake_run_demo(options: Any) -> None:
        assert options.dsn == "postgresql://localhost/demo"
        assert options.project_name == "demo-project"
        assert options.keep_project
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.artifact_root == tmp_path
        assert options.provider == "openai"
        assert options.model == "gpt-test"
        calls.append("run")

    monkeypatch.setattr(metadata_demo, "_run_demo", fake_run_demo)

    result = runner.invoke(
        metadata_demo.app,
        [
            "--dsn",
            "postgresql://localhost/demo",
            "--project-name",
            "demo-project",
            "--keep-project",
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--artifact-root",
            str(tmp_path),
            "--provider",
            "openai",
            "--model",
            "gpt-test",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]


def test_successful_lifecycle_requires_provider_response_and_success() -> None:
    metadata_demo = _load_metadata_demo()

    metadata_demo._verify_successful_lifecycle(
        _successful_events(), work_id="work-1"
    )


def test_successful_lifecycle_rejects_clean_failure() -> None:
    metadata_demo = _load_metadata_demo()

    try:
        metadata_demo._verify_successful_lifecycle(
            _failed_events(), work_id="work-1"
        )
    except RuntimeError as exc:
        assert "missing successful lifecycle events" in str(exc)
        assert "BillingError" in str(exc)
    else:
        raise AssertionError("clean failure lifecycle should not pass")


def _successful_events() -> list[EventEnvelope]:
    return [
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
        ),
        _event(
            StreamingLogEventType.attempt_succeeded,
            AttemptSucceededPayload(attempt=1),
        ),
        _event(
            StreamingLogEventType.work_completed,
            WorkCompletedPayload(status="succeeded", attempt=1),
        ),
    ]


def _failed_events() -> list[EventEnvelope]:
    return [
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
    ]


def _event(
    event_type: StreamingLogEventType,
    payload: Any,
) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"{event_type}-idem",
        payload=payload,
        work_id="work-1",
        attempt_id="attempt-1",
    )
