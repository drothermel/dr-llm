from __future__ import annotations

from typing import Any, cast

import pytest

from dr_llm.logging.events import get_generation_log_context
from dr_llm.pool.call_stats import CallStats
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pending.backend import (
    PoolPendingBackend,
    PoolPendingBackendConfig,
)
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.workers import ErrorDecision


class _FakeSchema:
    def __init__(self, *, name: str, key_column_names: list[str]) -> None:
        self.name = name
        self.key_column_names = key_column_names


class _FakePendingStore:
    def __init__(self) -> None:
        self.claimed: PendingSample | None = None
        self.last_claim_args: dict[str, Any] | None = None
        self.promote_result: PoolSample | None = PoolSample(key_values={"model": "m1"})
        self.promote_calls: list[dict[str, Any]] = []
        self.released: list[dict[str, str]] = []
        self.failed: list[dict[str, str]] = []
        self.release_result: bool = True
        self.fail_result: bool = True
        self.status_counts_value = PendingStatusCounts()

    def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        key_filter: PoolKeyFilter | None = None,
    ) -> PendingSample | None:
        self.last_claim_args = {
            "worker_id": worker_id,
            "lease_seconds": lease_seconds,
            "key_filter": key_filter,
        }
        return self.claimed

    def promote(
        self,
        *,
        pending_id: str,
        worker_id: str,
        payload: dict[str, Any] | None = None,
    ) -> PoolSample | None:
        self.promote_calls.append(
            {"pending_id": pending_id, "worker_id": worker_id, "payload": payload}
        )
        return self.promote_result

    def release_lease(self, *, pending_id: str, worker_id: str) -> bool:
        self.released.append({"pending_id": pending_id, "worker_id": worker_id})
        return self.release_result

    def fail(self, *, pending_id: str, worker_id: str, reason: str) -> bool:
        self.failed.append(
            {"pending_id": pending_id, "worker_id": worker_id, "reason": reason}
        )
        return self.fail_result

    def status_counts(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
    ) -> PendingStatusCounts:
        assert key_filter is None or isinstance(key_filter, PoolKeyFilter)
        return self.status_counts_value


class _FakeStore:
    def __init__(self) -> None:
        self.schema = _FakeSchema(name="pool-test", key_column_names=["model"])
        self.pending = _FakePendingStore()
        self.call_stats_inserts: list[CallStats] = []

    def insert_call_stats(self, stats: CallStats) -> None:
        self.call_stats_inserts.append(stats)


def _sample(*, attempt_count: int = 1) -> PendingSample:
    return PendingSample(
        pending_id="pending-1",
        key_values={"model": "m1"},
        sample_idx=0,
        attempt_count=attempt_count,
    )


def _eq_filter(**key_values: object) -> PoolKeyFilter:
    return PoolKeyFilter.eq(**key_values)


def test_backend_claims_one_item() -> None:
    store = _FakeStore()
    sample = _sample()
    store.pending.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(key_filter=_eq_filter(model="m1")),
    )

    claimed = backend.claim(worker_id="worker-1", lease_seconds=30)

    assert claimed == sample
    assert store.pending.last_claim_args == {
        "worker_id": "worker-1",
        "lease_seconds": 30,
        "key_filter": _eq_filter(model="m1"),
    }


def test_backend_claim_returns_none_when_store_is_empty() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    assert backend.claim(worker_id="worker-1", lease_seconds=30) is None


def test_backend_complete_promotes_pending_sample() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    result = {
        "text": "ok",
        "latency_ms": 500,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "finish_reason": "stop",
    }
    backend.complete(
        item=_sample(),
        result=result,
        worker_id="worker-1",
    )

    assert store.pending.promote_calls == [
        {
            "pending_id": "pending-1",
            "worker_id": "worker-1",
            "payload": result,
        }
    ]
    assert len(store.call_stats_inserts) == 1
    stats = store.call_stats_inserts[0]
    assert stats.latency_ms == 500
    assert stats.prompt_tokens == 10
    assert stats.attempt_count == 1
    assert stats.finish_reason == "stop"


def test_backend_complete_raises_when_promotion_fails() -> None:
    store = _FakeStore()
    store.pending.promote_result = None
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    with pytest.raises(RuntimeError, match="Failed to promote"):
        backend.complete(
            item=_sample(),
            result={"completion": "ok"},
            worker_id="worker-1",
        )


def test_backend_retries_while_attempts_are_within_limit() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )

    action = backend.handle_process_error(
        item=_sample(attempt_count=1),
        worker_id="worker-1",
        exc=RuntimeError("temporary"),
    )

    assert action == ErrorDecision.retry
    assert store.pending.released == [
        {"pending_id": "pending-1", "worker_id": "worker-1"}
    ]
    assert store.pending.failed == []


def test_backend_fails_when_retries_are_exhausted() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )

    action = backend.handle_process_error(
        item=_sample(attempt_count=2),
        worker_id="worker-1",
        exc=RuntimeError("boom"),
    )

    assert action == ErrorDecision.fail
    assert store.pending.failed == [
        {
            "pending_id": "pending-1",
            "worker_id": "worker-1",
            "reason": "RuntimeError: boom",
        }
    ]


def test_backend_retry_warns_when_release_lease_is_stale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    store.pending.release_result = False
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )

    with caplog.at_level("WARNING", logger="dr_llm.pool.pending.backend"):
        action = backend.handle_process_error(
            item=_sample(attempt_count=1),
            worker_id="worker-1",
            exc=RuntimeError("temporary"),
        )

    assert action == ErrorDecision.retry
    assert any("release_lease no-op" in r.message for r in caplog.records)


def test_backend_fail_warns_when_lease_is_stale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    store.pending.fail_result = False
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )

    with caplog.at_level("WARNING", logger="dr_llm.pool.pending.backend"):
        action = backend.handle_process_error(
            item=_sample(attempt_count=2),
            worker_id="worker-1",
            exc=RuntimeError("boom"),
        )

    assert action == ErrorDecision.fail
    assert any("fail no-op" in r.message for r in caplog.records)


def test_backend_snapshot_exposes_pool_specific_state() -> None:
    store = _FakeStore()
    store.pending.status_counts_value = PendingStatusCounts(pending=2, failed=1)
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(
            max_retries=3,
            key_filter=_eq_filter(model="m1"),
        ),
    )

    snapshot = backend.snapshot()

    assert snapshot.status_counts.pending == 2
    assert snapshot.status_counts.failed == 1
    assert snapshot.key_filter == _eq_filter(model="m1")
    assert snapshot.max_retries == 3


def test_backend_process_context_sets_pool_logging_context() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    with backend.process_context(item=_sample(), worker_id="worker-9"):
        context = get_generation_log_context()

    assert context["pool_name"] == "pool-test"
    assert context["worker_id"] == "worker-9"
