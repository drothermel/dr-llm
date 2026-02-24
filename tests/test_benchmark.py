from __future__ import annotations

from typing import Any
from uuid import uuid4

from llm_pool.benchmark import RepositoryBenchmarkConfig, run_repository_benchmark
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
    def __init__(self) -> None:
        self._calls: list[str] = []
        self._runs: dict[str, RunStatus] = {}

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
        cid = call_id or f"call_{uuid4().hex}"
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
        return self._calls[offset : offset + limit]

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionHandle:
        _ = metadata
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


def test_run_repository_benchmark_reports_expected_totals() -> None:
    repository = FakeRepository()

    stats = run_repository_benchmark(
        repository=repository,
        config=RepositoryBenchmarkConfig(workers=4, operations_per_worker=9),
    )

    assert stats.total_operations == 36
    assert stats.successful_operations == 36
    assert stats.failed_operations == 0
    assert stats.by_operation["record_call"] == 12
    assert stats.by_operation["session_roundtrip"] == 12
    assert stats.by_operation["read_calls"] == 12
    assert stats.failures_by_operation["record_call"] == 0
