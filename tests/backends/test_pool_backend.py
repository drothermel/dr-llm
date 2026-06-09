from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dr_llm.backends.converters import llm_response_to_backend_response
from dr_llm.backends.errors import BackendUnsupportedFeatureError
from dr_llm.backends.models import PoolBackendConfig
from dr_llm.backends.pool import PoolBackend
from dr_llm.llm import CallMode, LlmResponse, ProviderName, TokenUsage
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.sampling.acquisition import AcquireResult
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


def _pool_backend() -> tuple[PoolBackend, MagicMock, MagicMock, MagicMock]:
    store = MagicMock()
    store.bulk_load.return_value = []
    store.insert_sample.return_value = True
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
    assert response.text == "fresh"
    assert response.source == "generated"
    assert response.sample_id is not None


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


def test_pool_backend_acquire_rejects_unsupported_extensions() -> None:
    backend, _, _, _ = _pool_backend()
    with pytest.raises(BackendUnsupportedFeatureError):
        backend.acquire(
            make_backend_request(extensions={"tools": []}),
            "s1",
            n=1,
        )


@patch("dr_llm.backends.pool.catalog.load_schema", return_value=None)
@patch("dr_llm.backends.pool.PoolStore")
@patch("dr_llm.backends.pool.SamplingStore")
@patch("dr_llm.backends.pool.DbRuntime")
def test_pool_backend_init_sets_up_consumer(
    mock_runtime: MagicMock,
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
