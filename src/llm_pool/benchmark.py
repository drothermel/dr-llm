from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

KnownOperationName = Literal["record_call", "session_roundtrip", "read_calls"]
OperationName = Literal[
    "record_call", "session_roundtrip", "read_calls", "unknown_operation"
]
KnownPhaseName = Literal["warmup", "measured"]
PhaseName = Literal["warmup", "measured", "unknown_phase"]

_LATENCY_BINS_MS = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
    30000.0,
    60000.0,
)


class OperationMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_call: int = Field(default=1, ge=0)
    session_roundtrip: int = Field(default=1, ge=0)
    read_calls: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _validate_total_weight(self) -> OperationMix:
        if self.record_call + self.session_roundtrip + self.read_calls <= 0:
            raise ValueError("operation mix must include at least one positive weight")
        return self


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: int = Field(default=64, gt=0)
    total_operations: int = Field(default=20000, gt=0)
    warmup_operations: int | None = Field(default=None, ge=0)
    max_in_flight: int = Field(default=64, gt=0)
    run_type: str = "benchmark"
    operation_mix: OperationMix = Field(default_factory=OperationMix)
    artifact_path: str | None = None
    max_error_samples: int = Field(default=100, ge=1, le=1000)
    max_failure_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def resolved_warmup_operations(self) -> int:
        if self.warmup_operations is not None:
            return min(self.warmup_operations, self.total_operations)
        computed = max(100, int(self.total_operations * 0.05))
        return min(computed, self.total_operations)


class OperationStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    executed: int
    failed: int


class PhaseStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: KnownPhaseName
    total_operations: int
    successful_operations: int
    failed_operations: int
    elapsed_ms: int
    operations_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    by_operation: dict[KnownOperationName, OperationStats]


class BenchmarkErrorSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: PhaseName
    operation: OperationName
    message: str


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime
    config: BenchmarkConfig
    warmup: PhaseStats
    measured: PhaseStats
    errors_sampled: list[BenchmarkErrorSample]
    artifact_path: str


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

    def upsert_run_parameters(
        self, *, run_id: str, parameters: dict[str, Any]
    ) -> int: ...

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

    def record_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

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


class _LatencyHistogram:
    """Histogram of latencies in ms. Percentiles return bin upper-bounds (approximate)."""

    def __init__(self) -> None:
        self._counts = [0] * (len(_LATENCY_BINS_MS) + 1)
        self._total = 0
        self._max_ms: float = float("-inf")

    def observe(self, latency_ms: float) -> None:
        self._max_ms = max(self._max_ms, latency_ms)
        for idx, upper_bound in enumerate(_LATENCY_BINS_MS):
            if latency_ms <= upper_bound:
                self._counts[idx] += 1
                self._total += 1
                return
        self._counts[-1] += 1
        self._total += 1

    def merge(self, other: _LatencyHistogram) -> None:
        self._counts = [a + b for a, b in zip(self._counts, other._counts, strict=True)]
        self._total += other._total
        self._max_ms = max(self._max_ms, other._max_ms)

    def percentile(self, pct: float) -> float:
        # Percentiles return bin upper-bounds (approximate); overflow bucket uses actual max.
        if self._total == 0:
            return 0.0
        target = max(1, math.ceil((pct / 100.0) * self._total))
        running = 0
        for idx, count in enumerate(self._counts):
            running += count
            if running >= target:
                if idx < len(_LATENCY_BINS_MS):
                    return float(_LATENCY_BINS_MS[idx])
                return float(max(self._max_ms, _LATENCY_BINS_MS[-1]))
        return float(max(self._max_ms, _LATENCY_BINS_MS[-1]))


class _MutablePhaseStats:
    def __init__(self, *, max_error_samples: int) -> None:
        self._max_error_samples = max_error_samples
        self.total_operations = 0
        self.failed_operations = 0
        self.by_operation: dict[KnownOperationName, list[int]] = {
            "record_call": [0, 0],
            "session_roundtrip": [0, 0],
            "read_calls": [0, 0],
        }
        self.histogram = _LatencyHistogram()
        self.errors_sampled: list[BenchmarkErrorSample] = []

    def record(
        self,
        *,
        phase: KnownPhaseName,
        operation: KnownOperationName,
        elapsed_ms: float,
        failed: bool,
        error_text: str | None,
    ) -> None:
        self.total_operations += 1
        self.histogram.observe(elapsed_ms)
        op_counts = self.by_operation[operation]
        op_counts[0] += 1
        if failed:
            self.failed_operations += 1
            op_counts[1] += 1
            if error_text and len(self.errors_sampled) < self._max_error_samples:
                self.errors_sampled.append(
                    BenchmarkErrorSample(
                        phase=phase,
                        operation=operation,
                        message=error_text,
                    )
                )

    def merge(self, other: _MutablePhaseStats) -> None:
        self.total_operations += other.total_operations
        self.failed_operations += other.failed_operations
        self.histogram.merge(other.histogram)
        for operation in self.by_operation:
            self.by_operation[operation][0] += other.by_operation[operation][0]
            self.by_operation[operation][1] += other.by_operation[operation][1]
        remaining = self._max_error_samples - len(self.errors_sampled)
        if remaining > 0:
            self.errors_sampled.extend(other.errors_sampled[:remaining])

    def finalize(self, *, phase: KnownPhaseName, elapsed_ms: int) -> PhaseStats:
        successful_operations = self.total_operations - self.failed_operations
        ops_per_second = (
            self.total_operations / max(elapsed_ms / 1000.0, 0.001)
            if self.total_operations > 0
            else 0.0
        )
        by_operation: dict[KnownOperationName, OperationStats] = {
            operation: OperationStats(executed=counts[0], failed=counts[1])
            for operation, counts in self.by_operation.items()
        }
        return PhaseStats(
            phase=phase,
            total_operations=self.total_operations,
            successful_operations=successful_operations,
            failed_operations=self.failed_operations,
            elapsed_ms=elapsed_ms,
            operations_per_second=ops_per_second,
            p50_latency_ms=self.histogram.percentile(50.0),
            p95_latency_ms=self.histogram.percentile(95.0),
            by_operation=by_operation,
        )


class _OperationPlanner:
    def __init__(self, mix: OperationMix) -> None:
        self._thresholds: list[tuple[int, KnownOperationName]] = []
        running_total = 0
        for operation, weight in (
            ("record_call", mix.record_call),
            ("session_roundtrip", mix.session_roundtrip),
            ("read_calls", mix.read_calls),
        ):
            if weight <= 0:
                continue
            running_total += weight
            self._thresholds.append((running_total, operation))
        self._period = running_total

    def operation_for(self, index: int) -> KnownOperationName:
        slot = index % self._period
        for threshold, operation in self._thresholds:
            if slot < threshold:
                return operation
        return self._thresholds[-1][1]


def run_repository_benchmark(
    *,
    repository: RepositoryBenchmarkTarget,
    config: BenchmarkConfig,
) -> BenchmarkReport:
    repository.initialize()
    run_id = repository.start_run(
        run_type=config.run_type,
        status=RunStatus.running,
        metadata={"kind": "repository_benchmark"},
    )

    started_at = datetime.now(timezone.utc)
    warmup_phase = _empty_phase("warmup")
    measured_phase = _empty_phase("measured")
    errors_sampled: list[BenchmarkErrorSample] = []
    status = RunStatus.failed
    current_phase: PhaseName = "unknown_phase"

    artifact_path = _resolve_artifact_path(
        configured_path=config.artifact_path,
        run_id=run_id,
    )
    planner = _OperationPlanner(config.operation_mix)

    try:
        repository.upsert_run_parameters(
            run_id=run_id,
            parameters={
                "workers": config.workers,
                "total_operations": config.total_operations,
                "warmup_operations": config.resolved_warmup_operations,
                "max_in_flight": config.max_in_flight,
                "operation_mix": config.operation_mix.model_dump(mode="json"),
                "max_error_samples": config.max_error_samples,
                "max_failure_ratio": config.max_failure_ratio,
            },
        )

        current_phase = "warmup"
        warmup_phase, warmup_errors = _run_phase(
            repository=repository,
            run_id=run_id,
            planner=planner,
            phase="warmup",
            total_operations=config.resolved_warmup_operations,
            index_offset=0,
            workers=config.workers,
            max_in_flight=config.max_in_flight,
            max_error_samples=config.max_error_samples,
        )
        current_phase = "measured"
        measured_phase, measured_errors = _run_phase(
            repository=repository,
            run_id=run_id,
            planner=planner,
            phase="measured",
            total_operations=config.total_operations,
            index_offset=config.resolved_warmup_operations,
            workers=config.workers,
            max_in_flight=config.max_in_flight,
            max_error_samples=config.max_error_samples,
        )
        errors_sampled = (warmup_errors + measured_errors)[: config.max_error_samples]

        measured_failure_ratio = (
            measured_phase.failed_operations / measured_phase.total_operations
            if measured_phase.total_operations > 0
            else 0.0
        )
        status = (
            RunStatus.failed
            if measured_failure_ratio > config.max_failure_ratio
            else RunStatus.success
        )
    except Exception as exc:  # noqa: BLE001
        # The fatal exception may occur outside an operation context, so use a
        # sentinel operation value.
        errors_sampled = [
            BenchmarkErrorSample(
                phase=current_phase,
                operation="unknown_operation",
                message=f"fatal benchmark harness error: {type(exc).__name__}: {exc}",
            )
        ]
        status = RunStatus.failed

    finished_at = datetime.now(timezone.utc)
    report = BenchmarkReport(
        run_id=run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        warmup=warmup_phase,
        measured=measured_phase,
        errors_sampled=errors_sampled,
        artifact_path=str(artifact_path),
    )

    artifact_error: str | None = None
    try:
        _write_report(report=report, artifact_path=artifact_path)
        repository.record_artifact(
            run_id=run_id,
            artifact_type="benchmark_report",
            artifact_path=str(artifact_path),
            metadata={
                "status": report.status.value,
                "total_operations": report.measured.total_operations,
                "failed_operations": report.measured.failed_operations,
            },
        )
    except Exception as exc:  # noqa: BLE001
        artifact_error = f"{type(exc).__name__}: {exc}"
        status = RunStatus.failed

    repository.finish_run(
        run_id=run_id,
        status=status,
        metadata={
            "workers": config.workers,
            "total_operations": config.total_operations,
            "warmup_operations": config.resolved_warmup_operations,
            "failed_operations": report.measured.failed_operations,
            "operations_per_second": report.measured.operations_per_second,
            "p95_latency_ms": report.measured.p95_latency_ms,
            "artifact_path": str(artifact_path),
            "artifact_error": artifact_error,
        },
    )
    if status != report.status:
        report = report.model_copy(update={"status": status})
    return report


def _run_phase(
    *,
    repository: RepositoryBenchmarkTarget,
    run_id: str,
    planner: _OperationPlanner,
    phase: KnownPhaseName,
    total_operations: int,
    index_offset: int,
    workers: int,
    max_in_flight: int,
    max_error_samples: int,
) -> tuple[PhaseStats, list[BenchmarkErrorSample]]:
    if total_operations <= 0:
        empty = _empty_phase(phase)
        return empty, []

    next_index = 0
    index_lock = threading.Lock()
    semaphore = threading.Semaphore(max_in_flight)
    worker_results: list[_MutablePhaseStats | None] = [None] * workers
    phase_started = time.perf_counter()

    def worker_loop(worker_id: int) -> None:
        nonlocal next_index
        local = _MutablePhaseStats(max_error_samples=max_error_samples)
        while True:
            with index_lock:
                if next_index >= total_operations:
                    break
                op_index = next_index
                next_index += 1
            operation = planner.operation_for(index_offset + op_index)
            failed = False
            error_text: str | None = None
            try:
                with semaphore:
                    started = time.perf_counter()
                    _execute_operation(
                        repository=repository,
                        run_id=run_id,
                        operation=operation,
                        worker_id=worker_id,
                        op_index=index_offset + op_index,
                    )
            except Exception as exc:  # noqa: BLE001
                failed = True
                error_text = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            local.record(
                phase=phase,
                operation=operation,
                elapsed_ms=elapsed_ms,
                failed=failed,
                error_text=error_text,
            )
        worker_results[worker_id] = local

    threads = [
        threading.Thread(target=worker_loop, args=(worker_id,))
        for worker_id in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed_ms = int((time.perf_counter() - phase_started) * 1000)
    aggregate = _MutablePhaseStats(max_error_samples=max_error_samples)
    for local in worker_results:
        if local is not None:
            aggregate.merge(local)

    return aggregate.finalize(
        phase=phase, elapsed_ms=elapsed_ms
    ), aggregate.errors_sampled


def _execute_operation(
    *,
    repository: RepositoryBenchmarkTarget,
    run_id: str,
    operation: KnownOperationName,
    worker_id: int,
    op_index: int,
) -> None:
    if operation == "record_call":
        _record_call(
            repository=repository,
            run_id=run_id,
            worker_id=worker_id,
            op_index=op_index,
        )
        return
    if operation == "session_roundtrip":
        _session_roundtrip(
            repository=repository, worker_id=worker_id, op_index=op_index
        )
        return
    repository.list_calls(run_id=run_id, limit=10, offset=0)


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


def _resolve_artifact_path(*, configured_path: str | None, run_id: str) -> Path:
    if configured_path:
        return Path(configured_path)
    return Path(".llm_pool") / "benchmarks" / f"{run_id}.json"


def _write_report(*, report: BenchmarkReport, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            report.model_dump(mode="json", exclude_computed_fields=True),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _empty_phase(phase: KnownPhaseName) -> PhaseStats:
    by_operation: dict[KnownOperationName, OperationStats] = {
        "record_call": OperationStats(executed=0, failed=0),
        "session_roundtrip": OperationStats(executed=0, failed=0),
        "read_calls": OperationStats(executed=0, failed=0),
    }
    return PhaseStats(
        phase=phase,
        total_operations=0,
        successful_operations=0,
        failed_operations=0,
        elapsed_ms=0,
        operations_per_second=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
        by_operation=by_operation,
    )
