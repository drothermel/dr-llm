from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from dr_llm.backends.converters import llm_response_to_backend_response
from dr_llm.backends.errors import (
    BackendAcquireTimeoutError,
    BackendDrainTimeoutError,
    BackendGenerationError,
    BackendSchemaError,
    BackendUnsupportedFeatureError,
)
from dr_llm.sampling.errors import PoolTopupError
from dr_llm.backends.models import PoolBackendConfig
from dr_llm.backends.pool import PoolBackend
from dr_llm.llm import CallMode, LlmResponse, ProviderName, TokenUsage
from dr_llm.pool.backend import LlmPoolBackendState
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.sampling.acquisition import AcquireQuery, AcquireResult
from dr_llm.workers.models import WorkerSnapshot, WorkerStatCounts
from tests.backends._helpers import make_backend_request


def _llm_response(text: str = "ok") -> LlmResponse:
    return LlmResponse(
        text=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )


def _complete_sample(text: str = "ok") -> PoolSample:
    return PoolSample(
        key_values={"request_fingerprint": "fp"},
        request={
            "backend_request": make_backend_request().model_dump(mode="json"),
        },
        response=_llm_response(text).model_dump(mode="json"),
        finish_reason="stop",
        sample_idx=0,
    )


def _pool_backend() -> tuple[PoolBackend, MagicMock, MagicMock, MagicMock]:
    store = MagicMock()
    store.bulk_load.return_value = []
    store.insert_sample.return_value = True
    store.complete_count.return_value = 0
    service = MagicMock()
    sampling = MagicMock()
    runtime = MagicMock()

    backend = PoolBackend.__new__(PoolBackend)
    backend._config = PoolBackendConfig(pool_name="test_pool")
    backend._consumer_id = "test_pool"
    backend._registry = MagicMock()
    backend._direct = MagicMock()
    backend._runtime = runtime
    backend._schema = MagicMock()
    backend._store = store
    backend._sampling = sampling
    backend._service = service
    backend._drain_lock = threading.Lock()
    backend._closed = False
    return backend, store, service, backend._direct


def test_pool_backend_complete_uses_cache_hit() -> None:
    backend, store, _, direct = _pool_backend()
    cached = PoolSample(
        key_values={"request_fingerprint": "fp"},
        request={"backend_request": {}},
        response=_llm_response("cached").model_dump(mode="json"),
        finish_reason="stop",
        sample_idx=0,
        sample_id="cached-1",
    )
    store.bulk_load.return_value = [cached]

    response = backend.complete(make_backend_request())

    assert response.text == "cached"
    assert response.source == "pool_cache"
    assert response.sample_id == "cached-1"
    direct.complete.assert_not_called()
    store.insert_sample.assert_not_called()


def test_pool_backend_complete_generates_on_cache_miss() -> None:
    backend, store, _, direct = _pool_backend()
    direct.complete.return_value = llm_response_to_backend_response(
        _llm_response("fresh"),
        source="direct",
        fingerprint="fp",
    )

    response = backend.complete(make_backend_request())

    direct.complete.assert_called_once()
    store.insert_sample.assert_called_once()
    inserted_sample = store.insert_sample.call_args.args[0]
    assert "source" not in inserted_sample.response
    assert "sample_id" not in inserted_sample.response
    assert "request_fingerprint" not in inserted_sample.response
    assert response.text == "fresh"
    assert response.source == "generated"
    assert response.sample_id is not None


def test_pool_backend_acquire_generator_uses_direct_backend() -> None:
    backend, store, service, direct = _pool_backend()
    store.complete_count.side_effect = [0, 1]
    direct.complete.return_value = llm_response_to_backend_response(
        _llm_response("generated-via-direct"),
        source="direct",
        fingerprint="fp",
    )
    captured: list[PoolSample] = []

    def invoke_generator(
        query: AcquireQuery,
        *,
        consumer_id: str,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> AcquireResult:
        samples = generator_fn(query.key_values, query.n)
        captured.extend(samples)
        return AcquireResult(samples=samples)

    service.acquire_or_generate.side_effect = invoke_generator

    result = backend.acquire(make_backend_request(), "s1", n=1)

    direct.complete.assert_called_once()
    assert result.generated == 1
    assert result.responses[0].text == "generated-via-direct"
    assert captured
    stored_response = captured[0].response
    assert stored_response is not None
    assert "source" not in stored_response
    assert "sample_id" not in stored_response
    assert "request_fingerprint" not in stored_response


def test_pool_backend_acquire_delegates_to_service() -> None:
    backend, _, service, _ = _pool_backend()
    sample = PoolSample(
        key_values={"request_fingerprint": "fp"},
        request={"backend_request": {}},
        response=_llm_response().model_dump(mode="json"),
        finish_reason="stop",
        sample_idx=0,
    )
    service.acquire_or_generate.return_value = AcquireResult(samples=[sample])

    result = backend.acquire(make_backend_request(), "s1", n=1)

    assert len(result.responses) == 1
    assert result.claimed_from_cache == 1
    assert result.generated == 0
    service.acquire_or_generate.assert_called_once()


def test_pool_backend_acquire_rejects_blank_session_id() -> None:
    backend, _, service, _ = _pool_backend()

    with pytest.raises(ValueError, match="session_id"):
        backend.acquire(make_backend_request(), "  ", n=1)

    service.acquire_or_generate.assert_not_called()


def test_pool_backend_close_is_idempotent() -> None:
    backend, store, _, _ = _pool_backend()

    backend.close()
    backend.close()

    sampling = cast(MagicMock, backend._sampling)
    sampling.teardown_consumer.assert_called_once_with("test_pool")
    store.close.assert_called_once()


def test_pool_backend_acquire_maps_pool_topup_error() -> None:
    backend, _, service, _ = _pool_backend()
    service.acquire_or_generate.side_effect = PoolTopupError("top-up failed")

    with pytest.raises(
        BackendGenerationError, match="failed to generate samples"
    ) as exc_info:
        backend.acquire(make_backend_request(), "s1", n=1)

    assert isinstance(exc_info.value.__cause__, PoolTopupError)


def test_pool_backend_acquire_raises_timeout_during_generation() -> None:
    backend, store, service, direct = _pool_backend()
    backend._config = PoolBackendConfig(
        pool_name="test_pool",
        acquire_timeout_seconds=0.05,
    )

    def slow_complete(_request: object) -> object:
        time.sleep(0.2)
        return llm_response_to_backend_response(
            _llm_response("slow"),
            source="direct",
            fingerprint="fp",
        )

    direct.complete.side_effect = slow_complete

    def invoke_generator(
        query: AcquireQuery,
        *,
        consumer_id: str,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> AcquireResult:
        return AcquireResult(samples=generator_fn(query.key_values, query.n))

    service.acquire_or_generate.side_effect = invoke_generator
    store.complete_count.side_effect = [0, 0]

    with pytest.raises(BackendAcquireTimeoutError):
        backend.acquire(make_backend_request(), "s1", n=1)


def test_pool_backend_acquire_hybrid_cache_and_generated_accounting() -> None:
    backend, store, service, direct = _pool_backend()
    cached_samples = [_complete_sample(f"cache-{index}") for index in range(2)]
    direct.complete.return_value = llm_response_to_backend_response(
        _llm_response("fresh"),
        source="direct",
        fingerprint="fp",
    )
    iteration = 0

    def side_effect(
        query: AcquireQuery,
        *,
        consumer_id: str,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> AcquireResult:
        nonlocal iteration
        iteration += 1
        if iteration == 1:
            return AcquireResult(samples=cached_samples)
        samples = generator_fn(query.key_values, query.n)
        return AcquireResult(samples=samples)

    service.acquire_or_generate.side_effect = side_effect
    store.complete_count.side_effect = [10, 10, 10, 11]

    result = backend.acquire(make_backend_request(), "s1", n=3)

    assert result.claimed_from_cache == 2
    assert result.generated == 1
    assert [response.source for response in result.responses] == [
        "pool_cache",
        "pool_cache",
        "generated",
    ]


def test_pool_backend_acquire_accounting_when_insert_under_delivers() -> None:
    backend, store, service, _ = _pool_backend()
    samples = [_complete_sample(f"sample-{index}") for index in range(3)]
    service.acquire_or_generate.return_value = AcquireResult(samples=samples)
    store.complete_count.side_effect = [5, 7]

    result = backend.acquire(make_backend_request(), "s1", n=3)

    assert result.claimed_from_cache == 1
    assert result.generated == 2
    assert result.responses[0].source == "pool_cache"
    assert result.responses[1].source == "generated"
    assert result.responses[2].source == "generated"


def test_pool_backend_acquire_rejects_unsupported_extensions() -> None:
    backend, _, _, _ = _pool_backend()
    with pytest.raises(BackendUnsupportedFeatureError):
        backend.acquire(
            make_backend_request(extensions={"tools": []}),
            "s1",
            n=1,
        )


@patch("dr_llm.backends.pool.catalog.load_schema")
@patch("dr_llm.backends.pool.PoolStore")
@patch("dr_llm.backends.pool.SamplingStore")
@patch("dr_llm.backends.pool.DbRuntime")
def test_pool_backend_init_raises_schema_error_on_mismatched_keys(
    _mock_runtime: MagicMock,
    mock_sampling_store: MagicMock,
    mock_pool_store: MagicMock,
    mock_load_schema: MagicMock,
) -> None:
    existing_schema = MagicMock()
    existing_schema.key_column_names = ["legacy_key"]
    mock_load_schema.return_value = existing_schema

    with pytest.raises(BackendSchemaError, match="expected"):
        PoolBackend(
            PoolBackendConfig(
                pool_name="itest_backends", database_url="postgresql://x"
            ),
            registry=MagicMock(),
        )


@patch("dr_llm.backends.pool.catalog.load_schema", return_value=None)
@patch("dr_llm.backends.pool.PoolStore")
@patch("dr_llm.backends.pool.SamplingStore")
@patch("dr_llm.backends.pool.DbRuntime")
def test_pool_backend_init_sets_up_consumer(
    _mock_runtime: MagicMock,
    mock_sampling_store: MagicMock,
    mock_pool_store: MagicMock,
    _mock_load_schema: MagicMock,
) -> None:
    sampling = MagicMock()
    mock_sampling_store.from_pool_store.return_value = sampling

    backend = PoolBackend(
        PoolBackendConfig(
            pool_name="itest_backends", database_url="postgresql://x"
        ),
        registry=MagicMock(),
    )

    sampling.setup_consumer.assert_called_once_with("itest_backends")
    mock_pool_store.return_value.ensure_schema.assert_called_once()
    backend.close()
    sampling.teardown_consumer.assert_called_once_with("itest_backends")


def test_pool_backend_submit_batch_seeds_only_missing_fingerprints() -> None:
    backend, store, _, _ = _pool_backend()
    store.complete_count.return_value = 0
    store.incomplete_count.return_value = 0
    store.insert_sample.return_value = True

    result = backend.submit_batch(
        [make_backend_request(), make_backend_request(model="gpt-4.1")]
    )

    assert result.seeded == 2
    assert result.skipped == 0
    assert store.insert_sample.call_count == 2


def test_pool_backend_submit_batch_skips_existing_complete_cells() -> None:
    backend, store, _, _ = _pool_backend()
    store.complete_count.side_effect = [1, 0]
    store.incomplete_count.return_value = 0
    store.insert_sample.return_value = True

    result = backend.submit_batch(
        [make_backend_request(), make_backend_request(model="gpt-4.1")]
    )

    assert result.seeded == 1
    assert result.skipped == 1
    store.insert_sample.assert_called_once()


def test_pool_backend_submit_batch_skips_pending_incomplete_cells() -> None:
    backend, store, _, _ = _pool_backend()
    store.complete_count.return_value = 0
    store.incomplete_count.return_value = 1

    result = backend.submit_batch([make_backend_request()])

    assert result.seeded == 0
    assert result.skipped == 1
    store.insert_sample.assert_not_called()


@patch("dr_llm.backends.pool.start_workers")
def test_pool_backend_await_drain_is_single_flight(
    mock_start_workers: MagicMock,
) -> None:
    backend, _, _, _ = _pool_backend()
    backend._drain_lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    active_drains = 0
    max_active_drains = 0
    counter_lock = threading.Lock()

    def slow_start(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal active_drains, max_active_drains
        with counter_lock:
            active_drains += 1
            max_active_drains = max(max_active_drains, active_drains)
        started.set()
        release.wait(timeout=2)
        controller = MagicMock()
        snapshot = WorkerSnapshot(
            worker_count=1,
            counts=WorkerStatCounts(completed=1),
            backend_state=LlmPoolBackendState(incomplete=0, complete=1),
        )
        controller.snapshot.return_value = snapshot
        controller.join.return_value = snapshot
        with counter_lock:
            active_drains -= 1
        return controller

    mock_start_workers.side_effect = slow_start

    errors: list[Exception] = []

    def run_drain() -> None:
        try:
            backend.await_drain(timeout=5)
        except Exception as exc:
            errors.append(exc)

    thread_a = threading.Thread(target=run_drain)
    thread_b = threading.Thread(target=run_drain)
    thread_a.start()
    started.wait(timeout=2)
    thread_b.start()
    time.sleep(0.05)
    assert max_active_drains == 1
    release.set()
    thread_a.join()
    thread_b.join()
    assert not errors
    assert mock_start_workers.call_count == 2


@patch("dr_llm.backends.pool.start_workers")
def test_pool_backend_await_drain_returns_counts(
    mock_start_workers: MagicMock,
) -> None:
    backend, _, _, _ = _pool_backend()
    controller = MagicMock()
    snapshot = WorkerSnapshot(
        worker_count=1,
        counts=WorkerStatCounts(claimed=2, completed=2),
        backend_state=LlmPoolBackendState(incomplete=0, complete=2),
    )
    controller.snapshot.return_value = snapshot
    controller.join.return_value = snapshot
    mock_start_workers.return_value = controller

    result = backend.await_drain(timeout=5)

    assert result.incomplete == 0
    assert result.complete == 2
    assert result.worker_counts.completed == 2
    controller.stop.assert_called_once()


@patch("dr_llm.backends.pool.start_workers")
def test_pool_backend_await_drain_raises_on_timeout(
    mock_start_workers: MagicMock,
) -> None:
    backend, _, _, _ = _pool_backend()
    controller = MagicMock()
    busy_snapshot = WorkerSnapshot(
        worker_count=1,
        backend_state=LlmPoolBackendState(incomplete=3, complete=1),
    )
    controller.snapshot.return_value = busy_snapshot
    controller.join.return_value = busy_snapshot
    mock_start_workers.return_value = controller

    with pytest.raises(BackendDrainTimeoutError):
        backend.await_drain(timeout=0)
