"""Worker process function for backend pool fill."""

from __future__ import annotations

from dr_llm.backends.converters import backend_request_from_sample
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.response import LlmResponse
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.workers import ProcessFn

BACKEND_REQUEST_KEY = "backend_request"


def make_backend_process_fn(
    registry: ProviderRegistry,
    *,
    backend_request_key: str = BACKEND_REQUEST_KEY,
) -> ProcessFn[PoolSample, LlmResponse]:
    """Build a worker process function for backend-request pool samples."""

    def _process(sample: PoolSample) -> LlmResponse:
        if backend_request_key not in sample.request:
            msg = (
                f"PoolSample.request is missing key {backend_request_key!r}; "
                "was the pool seeded with backend_request payloads?"
            )
            raise KeyError(msg)
        request = backend_request_from_sample(sample)
        orchestrator = registry.get(request.provider)
        return orchestrator.generate(request.to_llm_request())

    return _process
