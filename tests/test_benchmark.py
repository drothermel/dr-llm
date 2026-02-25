from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from llm_pool.benchmark import BenchmarkConfig, OperationMix, run_repository_benchmark
from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    RunStatus,
    SessionHandle,
    SessionStatus,
    SessionTurnStatus,
    ToolPolicy,
)


class FakeRepository:
    def __init__(
        self,
        *,
        fail_operation: str | None = None,
        fail_upsert_run_parameters: bool = False,
        fail_record_artifact: bool = False,
    ) -> None:
        self._fail_operation = fail_operation
        self._fail_upsert_run_parameters = fail_upsert_run_parameters
        self._fail_record_artifact = fail_record_artifact
        self._calls: list[str] = []
        self._runs: dict[str, RunStatus] = {}
        self._artifacts: list[dict[str, Any]] = []
        self._parameters: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        return self._artifacts

    @property
    def run_parameters(self) -> dict[str, dict[str, Any]]:
        return self._parameters

    def initialize(self) -> None:
        return None

    def start_run(
        self,
        *,
        run_type: str = "generic",
        status: RunStatus = RunStatus.running,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        _ = run_type, metadata
        rid = run_id or f"run_{uuid4().hex}"
        self._runs[rid] = status
        return rid

    def upsert_run_parameters(self, *, run_id: str, parameters: dict[str, Any]) -> int:
        if self._fail_upsert_run_parameters:
            raise RuntimeError("upsert failure")
        self._parameters[run_id] = parameters
        return len(parameters)

    def finish_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _ = metadata
        self._runs[run_id] = status

    def record_call(
        self,
        *,
        request: LlmRequest,
        response: LlmResponse | None = None,
        run_id: str | None = None,
        status: str | None = None,
        mode: CallMode | str | None = None,
        error_text: str | None = None,
        external_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> str:
        _ = (
            request,
            response,
            run_id,
            status,
            mode,
            error_text,
            external_call_id,
            metadata,
        )
        if self._fail_operation == "record_call":
            raise RuntimeError("record_call failure")
        cid = call_id or f"call_{uuid4().hex}"
        with self._lock:
            self._calls.append(cid)
        return cid

    def list_calls(
        self,
        *,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        _ = run_id
        if self._fail_operation == "read_calls":
            raise RuntimeError("read_calls failure")
        with self._lock:
            return self._calls[offset : offset + limit]

    def record_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self._fail_record_artifact:
            raise RuntimeError("record_artifact failure")
        self._artifacts.append(
            {
                "run_id": run_id,
                "artifact_type": artifact_type,
                "artifact_path": artifact_path,
                "metadata": metadata,
            }
        )
        return f"artifact_{uuid4().hex}"

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionHandle:
        _ = metadata
        if self._fail_operation == "session_roundtrip":
            raise RuntimeError("session_roundtrip failure")
        return SessionHandle(
            session_id=session_id or f"session_{uuid4().hex}",
            status=SessionStatus.active,
            version=1,
            strategy_mode=strategy_mode,
        )

    def create_session_turn(
        self,
        *,
        session_id: str,
        status: SessionTurnStatus = SessionTurnStatus.active,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        _ = session_id, status, metadata
        return (f"turn_{uuid4().hex}", 1)

    def append_session_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        _ = session_id, event_type, payload, turn_id
        return event_id or f"event_{uuid4().hex}"

    def complete_session_turn(
        self,
        *,
        turn_id: str,
        status: SessionTurnStatus,
    ) -> None:
        _ = turn_id, status

    def update_session_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        last_error_text: str | None = None,
    ) -> None:
        _ = session_id, status, last_error_text


def test_benchmark_phase_and_operation_distribution(tmp_path: Path) -> None:
    repository = FakeRepository()
    artifact_path = tmp_path / "bench-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=4,
            total_operations=30,
            warmup_operations=6,
            max_in_flight=4,
            operation_mix=OperationMix(record_call=2, session_roundtrip=1, read_calls=1),
            artifact_path=str(artifact_path),
        ),
    )

    assert report.status == RunStatus.success
    assert report.warmup.total_operations == 6
    assert report.measured.total_operations == 30
    assert report.measured.failed_operations == 0
    assert report.measured.by_operation["record_call"].executed == 14
    assert report.measured.by_operation["session_roundtrip"].executed == 8
    assert report.measured.by_operation["read_calls"].executed == 8
    assert artifact_path.exists()


def test_benchmark_allows_max_in_flight_lower_than_workers(tmp_path: Path) -> None:
    repository = FakeRepository()
    artifact_path = tmp_path / "throttled-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=8,
            total_operations=120,
            warmup_operations=12,
            max_in_flight=2,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.status == RunStatus.success
    assert report.measured.total_operations == 120
    assert artifact_path.exists()


def test_benchmark_failure_ratio_marks_run_failed(tmp_path: Path) -> None:
    repository = FakeRepository(fail_operation="read_calls")
    artifact_path = tmp_path / "bench-failed.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=2,
            total_operations=50,
            warmup_operations=0,
            max_in_flight=2,
            operation_mix=OperationMix(record_call=0, session_roundtrip=0, read_calls=1),
            max_failure_ratio=0.2,
            artifact_path=str(artifact_path),
            max_error_samples=5,
        ),
    )

    assert report.status == RunStatus.failed
    assert report.measured.failed_operations == 50
    assert len(report.errors_sampled) == 5
    assert artifact_path.exists()


def test_benchmark_fatal_error_uses_unknown_operation_sentinel(tmp_path: Path) -> None:
    repository = FakeRepository(fail_upsert_run_parameters=True)
    artifact_path = tmp_path / "fatal-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=2,
            total_operations=10,
            warmup_operations=0,
            max_in_flight=1,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.status == RunStatus.failed
    assert report.errors_sampled[0].phase == "unknown_phase"
    assert report.errors_sampled[0].operation == "unknown_operation"
    assert artifact_path.exists()


def test_benchmark_records_artifact_and_parameters(tmp_path: Path) -> None:
    repository = FakeRepository()
    artifact_path = tmp_path / "report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=3,
            total_operations=12,
            warmup_operations=3,
            max_in_flight=3,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.artifact_path == str(artifact_path)
    assert repository.artifacts
    assert repository.artifacts[0]["artifact_type"] == "benchmark_report"
    assert repository.artifacts[0]["artifact_path"] == str(artifact_path)
    assert report.run_id in repository.run_parameters


def test_benchmark_report_file_reflects_failed_status_after_artifact_error(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(fail_record_artifact=True)
    artifact_path = tmp_path / "artifact-error-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=3,
            total_operations=12,
            warmup_operations=0,
            max_in_flight=3,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.status == RunStatus.failed
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert persisted["status"] == RunStatus.failed.value


def test_benchmark_large_operation_count_completes(tmp_path: Path) -> None:
    repository = FakeRepository()
    artifact_path = tmp_path / "large-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=8,
            total_operations=20000,
            warmup_operations=1000,
            max_in_flight=8,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.status == RunStatus.success
    assert report.measured.total_operations == 20000
    assert report.measured.p95_latency_ms >= report.measured.p50_latency_ms
    assert artifact_path.exists()
