from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from dr_llm.backends.async_bridge import run_in_thread
from dr_llm.backends.direct import DirectBackend
from dr_llm.backends.models import AcquireResult, DrainResult
from dr_llm.backends.pool import PoolBackend
from dr_llm.workers.models import WorkerStatCounts
from tests.backends._helpers import make_backend_request


def test_run_in_thread_executes_callable() -> None:
    result = asyncio.run(run_in_thread(lambda: 42))
    assert result == 42


def test_direct_backend_acomplete_uses_thread_bridge() -> None:
    backend = DirectBackend(registry=MagicMock())
    expected = MagicMock()
    with patch.object(
        backend,
        "complete",
        return_value=expected,
    ) as complete:
        result = asyncio.run(backend.acomplete(make_backend_request()))
    complete.assert_called_once()
    assert result is expected


def test_pool_backend_acomplete_uses_thread_bridge() -> None:
    backend = PoolBackend.__new__(PoolBackend)
    expected = MagicMock()
    backend.complete = MagicMock(return_value=expected)
    result = asyncio.run(backend.acomplete(make_backend_request()))
    backend.complete.assert_called_once()
    assert result is expected


def test_pool_backend_aacquire_uses_thread_bridge() -> None:
    backend = PoolBackend.__new__(PoolBackend)
    expected = AcquireResult()
    backend.acquire = MagicMock(return_value=expected)
    result = asyncio.run(backend.aacquire(make_backend_request(), "s1", 2))
    backend.acquire.assert_called_once_with(
        make_backend_request(),
        "s1",
        2,
        request_id=None,
    )
    assert result is expected


def test_pool_backend_adrain_uses_thread_bridge() -> None:
    backend = PoolBackend.__new__(PoolBackend)
    expected = DrainResult(
        incomplete=0,
        complete=3,
        worker_counts=WorkerStatCounts(completed=3),
    )
    backend.await_drain = MagicMock(return_value=expected)
    result = asyncio.run(backend.adrain(timeout=10))
    backend.await_drain.assert_called_once_with(10)
    assert result is expected
