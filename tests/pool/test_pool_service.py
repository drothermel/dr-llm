from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from dr_llm.pool.models import AcquireQuery, AcquireResult
from dr_llm.pool.pool_service import PoolService


def test_wait_and_reacquire_does_not_consult_in_flight_count_when_rows_were_bumped(
    monkeypatch,
) -> None:
    store = MagicMock()
    store.pending = SimpleNamespace(
        bump_priority=MagicMock(return_value=1),
        count_in_flight=MagicMock(side_effect=AssertionError("should not be called")),
    )
    store.acquire = MagicMock(return_value=AcquireResult())
    service = PoolService(store, pending_poll_interval_s=0.05, pending_poll_timeout_s=0.02)
    query = AcquireQuery(run_id="run-1", key_values={"dim_a": "svc", "dim_b": 1}, n=1)

    monotonic_values = iter([0.0, 0.0, 0.03])
    monkeypatch.setattr("dr_llm.pool.pool_service.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("dr_llm.pool.pool_service.time.sleep", lambda _seconds: None)

    result = service._wait_and_reacquire(query, AcquireResult())

    assert result.claimed == 0
    assert store.pending.bump_priority.call_count == 1
    assert store.acquire.call_count == 1

