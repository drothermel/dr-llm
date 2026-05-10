from __future__ import annotations

from typing import Any, cast

from dr_llm.logging.events import get_generation_log_context
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pending.backend import (
    PoolPendingBackend,
    PoolPendingBackendConfig,
)
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.workers import ErrorDecision


class _FakeSchema:
    def __init__(self, *, name: str, key_column_names: list[str]) -> None:
        self.name = name
        self.key_column_names = key_column_names


class _FakeStore:
    def __init__(self) -> None:
        self.schema = _FakeSchema(name="pool-test", key_column_names=["model"])
        self.claimed: PoolSample | None = None
        self.last_claim_args: dict[str, Any] | None = None
        self.complete_calls: list[dict[str, Any]] = []
        self.release_calls: list[dict[str, str]] = []
        self.calls: list[str] = []
        self.complete_result: bool = True
        self.release_result: bool = True
        self.incomplete = 0
        self.complete = 0

    def claim_lease(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        key_filter: PoolKeyFilter | None = None,
    ) -> PoolSample | None:
        self.last_claim_args = {
            "worker_id": worker_id,
            "lease_seconds": lease_seconds,
            "key_filter": key_filter,
        }
        return self.claimed

    def complete_sample(
        self,
        *,
        sample_id: str,
        response: dict[str, Any],
        finish_reason: str | None,
        attempt_count: int,
    ) -> bool:
        self.calls.append("complete_sample")
        self.complete_calls.append(
            {
                "sample_id": sample_id,
                "response": response,
                "finish_reason": finish_reason,
                "attempt_count": attempt_count,
            }
        )
        return self.complete_result

    def release_lease(self, *, sample_id: str, worker_id: str) -> bool:
        self.calls.append("release_lease")
        self.release_calls.append({"sample_id": sample_id, "worker_id": worker_id})
        return self.release_result

    def incomplete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        assert key_filter is None or isinstance(key_filter, PoolKeyFilter)
        return self.incomplete

    def complete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        assert key_filter is None or isinstance(key_filter, PoolKeyFilter)
        return self.complete


def _sample(*, sample_id: str = "sample-1") -> PoolSample:
    return PoolSample(
        sample_id=sample_id,
        key_values={"model": "m1"},
        sample_idx=0,
    )


def _eq_filter(**key_values: object) -> PoolKeyFilter:
    return PoolKeyFilter.eq(**key_values)


def test_backend_claims_one_item_and_tracks_attempt() -> None:
    store = _FakeStore()
    sample = _sample()
    store.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(key_filter=_eq_filter(model="m1")),
    )

    claimed = backend.claim(worker_id="worker-1", lease_seconds=30)

    assert claimed == sample
    assert store.last_claim_args == {
        "worker_id": "worker-1",
        "lease_seconds": 30,
        "key_filter": _eq_filter(model="m1"),
    }
    backend.complete(item=sample, result={"text": "ok"}, worker_id="worker-1")
    assert store.complete_calls[-1]["attempt_count"] == 1


def test_backend_claim_returns_none_when_store_is_empty() -> None:
    store = _FakeStore()
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )

    assert backend.claim(worker_id="worker-1", lease_seconds=30) is None


def test_backend_complete_writes_response_then_releases_lease() -> None:
    store = _FakeStore()
    sample = _sample()
    store.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(),
    )
    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample
    result = {"text": "ok", "finish_reason": "stop"}

    backend.complete(
        item=sample,
        result=result,
        worker_id="worker-1",
    )

    assert store.calls == ["complete_sample", "release_lease"]
    assert store.complete_calls == [
        {
            "sample_id": "sample-1",
            "response": result,
            "finish_reason": "stop",
            "attempt_count": 1,
        }
    ]
    assert store.release_calls == [{"sample_id": "sample-1", "worker_id": "worker-1"}]


def test_backend_retries_while_attempts_are_within_limit() -> None:
    store = _FakeStore()
    sample = _sample()
    store.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )
    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample

    action = backend.handle_process_error(
        item=sample,
        worker_id="worker-1",
        exc=RuntimeError("temporary"),
    )

    assert action == ErrorDecision.retry
    assert store.complete_calls == []
    assert store.release_calls == [{"sample_id": "sample-1", "worker_id": "worker-1"}]


def test_backend_fails_when_retries_are_exhausted() -> None:
    store = _FakeStore()
    sample = _sample()
    store.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=1),
    )
    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample
    assert (
        backend.handle_process_error(
            item=sample,
            worker_id="worker-1",
            exc=RuntimeError("temporary"),
        )
        == ErrorDecision.retry
    )
    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample

    action = backend.handle_process_error(
        item=sample,
        worker_id="worker-1",
        exc=RuntimeError("boom"),
    )

    assert action == ErrorDecision.fail
    assert store.calls == [
        "release_lease",
        "complete_sample",
        "release_lease",
    ]
    assert store.complete_calls == [
        {
            "sample_id": "sample-1",
            "response": {"error": "RuntimeError: boom"},
            "finish_reason": "error",
            "attempt_count": 2,
        }
    ]


def test_reclaiming_same_sample_id_increments_attempt_counter() -> None:
    store = _FakeStore()
    sample = _sample()
    store.claimed = sample
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(max_retries=2),
    )

    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample
    assert (
        backend.handle_process_error(
            item=sample,
            worker_id="worker-1",
            exc=RuntimeError("temporary"),
        )
        == ErrorDecision.retry
    )
    assert backend.claim(worker_id="worker-1", lease_seconds=30) == sample
    backend.complete(item=sample, result={"text": "ok"}, worker_id="worker-1")

    assert store.complete_calls[-1]["attempt_count"] == 2


def test_backend_snapshot_exposes_pool_specific_state() -> None:
    store = _FakeStore()
    store.incomplete = 2
    store.complete = 5
    backend = PoolPendingBackend(
        cast(PoolStore, store),
        config=PoolPendingBackendConfig(
            max_retries=3,
            key_filter=_eq_filter(model="m1"),
        ),
    )

    snapshot = backend.snapshot()

    assert snapshot.incomplete == 2
    assert snapshot.complete == 5
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
