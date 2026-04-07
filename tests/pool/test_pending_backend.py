from __future__ import annotations

from typing import Any, cast

import pytest

from dr_llm.logging.events import get_generation_log_context
from dr_llm.pool.pending.backend import (
    PoolPendingBackend,
    PoolPendingBackendConfig,
)
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.sample_store import PoolStore
from dr_llm.workers import ErrorDecision


class _FakeSchema:
    def __init__(self, *, name: str, key_column_names: list[str]) -> None:
        self.name = name
        self.key_column_names = key_column_names


class _FakePendingStore:
    def __init__(self) -> None:
        self.claimed: list[PendingSample] = []
        self.last_claim_args: dict[str, Any] | None = None
        self.promote_result: PoolSample | None = PoolSample(key_values={"model": "m1"})
        self.promote_calls: list[dict[str, Any]] = []
        self.released: list[dict[str, str]] = []
        self.failed: list[dict[str, str]] = []
        self.status_counts_value = PendingStatusCounts()

    def claim_pending(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        key_filter: dict[str, Any] | None = None,
    ) -> list[PendingSample]:
        self.last_claim_args = {
            "worker_id": worker_id,
            "limit": limit,
            "lease_seconds": lease_seconds,
            "key_filter": key_filter,
        }
        return list(self.claimed)

    def promote_pending(
        self,
        *,
        pending_id: str,
        payload: dict[str, Any] | None = None,
    ) -> PoolSample | None:
        self.promote_calls.append({"pending_id": pending_id, "payload": payload})
        return self.promote_result

    def release_pending_lease(self, *, pending_id: str, worker_id: str) -> None:
        self.released.append({"pending_id": pending_id, "worker_id": worker_id})

    def fail_pending(self, *, pending_id: str, worker_id: str, reason: str) -> None:
        self.failed.append(
            {"pending_id": pending_id, "worker_id": worker_id, "reason": reason}
        )

    def status_counts(
        self,
        *,
        key_filter: dict[str, Any] | None = None,
    ) -> PendingStatusCounts:
        assert key_filter is None or isinstance(key_filter, dict)
        return self.status_counts_value


class _FakeStore:
    def __init__(self) -> None:
        self.schema = _FakeSchema(name="pool-test", key_column_names=["model"])
        self.pending = _FakePendingStore()
        self.init_calls = 0

    def init_schema(self) -> None:
        self.init_calls += 1


def _sample(*, attempt_count: int = 1) -> PendingSample:
    return PendingSample(
        pending_id="pending-1",
        key_values={"model": "m1"},
        sample_idx=0,
        attempt_count=attempt_count,
    )


def test_backend_initializes_store_and_claims_one_item() -> None:
    store = _FakeStore()
    store.pending.claimed = [_sample()]
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(key_filter={"model": "m1"}),
    )

    claimed = backend.claim(worker_id="worker-1", lease_seconds=30)

    assert store.init_calls == 1
    assert claimed == store.pending.claimed
    assert store.pending.last_claim_args == {
        "worker_id": "worker-1",
        "limit": 1,
        "lease_seconds": 30,
        "key_filter": {"model": "m1"},
    }


def test_backend_claim_raises_if_store_returns_multiple_items() -> None:
    store = _FakeStore()
    store.pending.claimed = [_sample(), _sample(attempt_count=2)]
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    with pytest.raises(RuntimeError, match="more than one item"):
        backend.claim(worker_id="worker-1", lease_seconds=30)


def test_backend_complete_promotes_pending_sample() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    backend.complete(
        item=_sample(),
        result={"completion": "ok"},
        worker_id="worker-1",
    )

    assert store.pending.promote_calls == [
        {"pending_id": "pending-1", "payload": {"completion": "ok"}}
    ]


def test_backend_complete_rejects_non_dict_payload() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    with pytest.raises(TypeError, match="payload dict"):
        backend.complete(
            item=_sample(),
            result=cast(dict[str, Any], ["not-a-dict"]),
            worker_id="worker-1",
        )


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


def test_backend_snapshot_exposes_pool_specific_state() -> None:
    store = _FakeStore()
    store.pending.status_counts_value = PendingStatusCounts(promoted=2, failed=1)
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(
            max_retries=3,
            key_filter={"model": "m1"},
        ),
    )

    snapshot = backend.snapshot()

    assert snapshot.status_counts.promoted == 2
    assert snapshot.status_counts.failed == 1
    assert snapshot.key_filter == {"model": "m1"}
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
