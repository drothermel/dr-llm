from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    RunStatus,
    SessionStatus,
    SessionTurnStatus,
    TokenUsage,
    ToolPolicy,
)

OperationName = Literal["record_call", "session_roundtrip", "read_calls"]


class RepositoryBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: int = Field(default=64, gt=0)
    operations_per_worker: int = Field(default=100, gt=0)
    run_type: str = "benchmark"


class RepositoryBenchmarkStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: int
    operations_per_worker: int
    total_operations: int
    successful_operations: int
    failed_operations: int
    elapsed_ms: int
    operations_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    by_operation: dict[OperationName, int]
    failures_by_operation: dict[OperationName, int]


class RepositoryBenchmarkTarget(Protocol):
    def initialize(self) -> None: ...

    def start_run(
        self,
        *,
        run_type: str = "generic",
        status: RunStatus = RunStatus.running,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str: ...

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
    ) -> str: ...

    def finish_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def list_calls(
        self,
        *,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]: ...

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Any: ...

    def create_session_turn(
        self,
        *,
        session_id: str,
        status: SessionTurnStatus = SessionTurnStatus.active,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]: ...

    def append_session_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        event_id: str | None = None,
    ) -> str: ...

    def complete_session_turn(
        self,
        *,
        turn_id: str,
        status: SessionTurnStatus,
    ) -> None: ...

    def update_session_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        last_error_text: str | None = None,
    ) -> None: ...


def run_repository_benchmark(
    *,
    repository: RepositoryBenchmarkTarget,
    config: RepositoryBenchmarkConfig,
) -> RepositoryBenchmarkStats:
    repository.initialize()
    run_id = repository.start_run(
        run_type=config.run_type,
        status=RunStatus.running,
        metadata={"kind": "repository_benchmark"},
    )

    totals: dict[OperationName, int] = {
        "record_call": 0,
        "session_roundtrip": 0,
        "read_calls": 0,
    }
    failures: dict[OperationName, int] = {
        "record_call": 0,
        "session_roundtrip": 0,
        "read_calls": 0,
    }
    latencies_ms: list[float] = []
    lock = threading.Lock()

    def do_operation(worker_id: int, op_index: int) -> None:
        op_kind = _operation_for(op_index)
        started = time.perf_counter()
        failed = False
        try:
            if op_kind == "record_call":
                _record_call(
                    repository=repository,
                    run_id=run_id,
                    worker_id=worker_id,
                    op_index=op_index,
                )
            elif op_kind == "session_roundtrip":
                _session_roundtrip(
                    repository=repository, worker_id=worker_id, op_index=op_index
                )
            else:
                repository.list_calls(run_id=run_id, limit=10, offset=0)
        except Exception:  # noqa: BLE001
            failed = True
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with lock:
                totals[op_kind] += 1
                if failed:
                    failures[op_kind] += 1
                latencies_ms.append(elapsed_ms)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = []
        for worker_id in range(config.workers):
            for op_index in range(config.operations_per_worker):
                futures.append(pool.submit(do_operation, worker_id, op_index))
        for future in futures:
            future.result()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    total_ops = sum(totals.values())
    failed_ops = sum(failures.values())
    successful_ops = total_ops - failed_ops
    ops_per_second = (total_ops / max(elapsed_ms / 1000.0, 0.001)) if total_ops else 0.0

    repository.finish_run(
        run_id=run_id,
        status=RunStatus.success if failed_ops == 0 else RunStatus.failed,
        metadata={
            "workers": config.workers,
            "operations_per_worker": config.operations_per_worker,
            "total_operations": total_ops,
            "failed_operations": failed_ops,
        },
    )

    return RepositoryBenchmarkStats(
        workers=config.workers,
        operations_per_worker=config.operations_per_worker,
        total_operations=total_ops,
        successful_operations=successful_ops,
        failed_operations=failed_ops,
        elapsed_ms=elapsed_ms,
        operations_per_second=ops_per_second,
        p50_latency_ms=_percentile(latencies_ms, 50.0),
        p95_latency_ms=_percentile(latencies_ms, 95.0),
        by_operation=totals,
        failures_by_operation=failures,
    )


def _record_call(
    *,
    repository: RepositoryBenchmarkTarget,
    run_id: str,
    worker_id: int,
    op_index: int,
) -> None:
    request = LlmRequest(
        provider="openai",
        model="gpt-4.1-mini",
        messages=[
            Message(role="user", content=f"bench worker={worker_id} op={op_index}")
        ],
        metadata={"benchmark": True, "worker_id": worker_id, "op_index": op_index},
    )
    response = LlmResponse(
        text="ok",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3),
        raw_json={"benchmark": True},
        latency_ms=5,
        provider="openai",
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )
    repository.record_call(
        request=request,
        response=response,
        run_id=run_id,
        metadata={"operation": "record_call"},
    )


def _session_roundtrip(
    *,
    repository: RepositoryBenchmarkTarget,
    worker_id: int,
    op_index: int,
) -> None:
    handle = repository.start_session(
        strategy_mode=ToolPolicy.native_preferred,
        metadata={
            "benchmark": True,
            "worker_id": worker_id,
            "op_index": op_index,
            "provider": "openai",
            "model": "gpt-4.1-mini",
        },
        session_id=f"bench_session_{uuid4().hex}",
    )
    turn_id, _ = repository.create_session_turn(
        session_id=handle.session_id,
        status=SessionTurnStatus.active,
        metadata={"benchmark": True},
    )
    repository.append_session_event(
        session_id=handle.session_id,
        turn_id=turn_id,
        event_type="benchmark.event",
        payload={"worker_id": worker_id, "op_index": op_index},
    )
    repository.complete_session_turn(
        turn_id=turn_id, status=SessionTurnStatus.completed
    )
    repository.update_session_status(
        session_id=handle.session_id,
        status=SessionStatus.completed,
    )


def _operation_for(op_index: int) -> OperationName:
    return ("record_call", "session_roundtrip", "read_calls")[op_index % 3]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return float(ordered[idx])
