"""Pool-backed backend with cache, session acquire, and batch fill."""

from __future__ import annotations

import time
from collections.abc import Callable
from os import getenv
from typing import Any
from uuid import uuid4

from dr_llm.backends.async_bridge import run_in_thread
from dr_llm.backends.converters import (
    backend_request_payload,
    pool_sample_to_backend_response,
)
from dr_llm.backends.direct import DirectBackend
from dr_llm.backends.errors import (
    BackendAcquireTimeoutError,
    BackendDrainTimeoutError,
    BackendGenerationError,
    BackendSchemaError,
)
from dr_llm.backends.process_fn import make_backend_process_fn
from dr_llm.backends.fingerprint import fingerprint_request
from dr_llm.backends.models import (
    AcquireResult,
    BackendRequest,
    BackendResponse,
    DrainResult,
    PoolBackendConfig,
    SubmitResult,
)
from dr_llm.backends.schema import BACKENDS_KEY_COLUMN, backends_pool_schema
from dr_llm.backends.validation import validate_v1_request
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.default_registry import build_default_registry
from dr_llm.pool.db import catalog
from dr_llm.pool.backend import LlmPoolBackend, LlmPoolBackendConfig
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.pool.progress import pool_is_idle
from dr_llm.workers import WorkerConfig, start_workers
from dr_llm.workers.worker_controller import WorkerController
from dr_llm.sampling.acquisition import AcquireQuery
from dr_llm.sampling.errors import PoolTopupError
from dr_llm.sampling.pool_service import PoolService
from dr_llm.sampling.sampling_store import SamplingStore

_GENERATOR_FN = Callable[[dict[str, Any], int], list[PoolSample]]


class PoolBackend:
    """Fingerprint-keyed pool backend with cache and session acquire."""

    def __init__(
        self,
        config: PoolBackendConfig,
        *,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._config = config
        self._consumer_id = config.consumer_id or config.pool_name
        self._registry = registry or build_default_registry()
        self._direct = DirectBackend(self._registry)

        database_url = config.database_url or getenv(
            "DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm"
        )
        self._runtime = DbRuntime(
            DbConfig(
                dsn=database_url,
                application_name=f"dr_llm_backends_{config.pool_name}",
            )
        )
        self._schema = backends_pool_schema(config.pool_name)
        self._store = PoolStore(self._schema, self._runtime)
        self._store.ensure_schema()
        self._validate_existing_schema()
        self._sampling = SamplingStore.from_pool_store(self._store)
        self._sampling.setup_consumer(self._consumer_id)
        self._service = PoolService(self._store, sampling_store=self._sampling)

    @property
    def store(self) -> PoolStore:
        return self._store

    def close(self) -> None:
        self._sampling.teardown_consumer(self._consumer_id)
        self._store.close()

    def complete(self, request: BackendRequest) -> BackendResponse:
        validate_v1_request(request)
        fingerprint = fingerprint_request(request)
        cached = self._first_complete_sample(fingerprint)
        if cached is not None:
            return pool_sample_to_backend_response(
                cached,
                source="pool_cache",
                fingerprint=fingerprint,
            )

        generated = self._direct.complete(request)
        sample = self._backend_response_to_pool_sample(
            generated,
            request,
            {BACKENDS_KEY_COLUMN: fingerprint},
        )
        self._store.insert_sample(sample)
        return generated.model_copy(
            update={
                "source": "generated",
                "sample_id": sample.sample_id,
                "request_fingerprint": fingerprint,
            }
        )

    def acquire(
        self,
        request: BackendRequest,
        session_id: str,
        n: int,
        *,
        request_id: str | None = None,
    ) -> AcquireResult:
        validate_v1_request(request)
        if n < 0:
            msg = f"acquire count must be non-negative, got {n}"
            raise ValueError(msg)

        fingerprint = fingerprint_request(request)
        key_values = {BACKENDS_KEY_COLUMN: fingerprint}
        responses: list[BackendResponse] = []
        claimed_from_cache = 0
        generated = 0
        deadline = time.monotonic() + self._config.acquire_timeout_seconds
        resolved_request_id = request_id or uuid4().hex

        while len(responses) < n:
            if time.monotonic() >= deadline:
                raise BackendAcquireTimeoutError(
                    f"acquired {len(responses)} of {n} samples for session "
                    f"{session_id!r} before timeout"
                )

            remaining = n - len(responses)
            generated_before = generated

            def generator_fn(
                cell_key_values: dict[str, Any],
                deficit: int,
            ) -> list[PoolSample]:
                nonlocal generated
                created = self._generate_completed_samples(
                    request,
                    cell_key_values,
                    deficit,
                )
                generated += len(created)
                return created

            query = AcquireQuery(
                run_id=session_id,
                request_id=resolved_request_id,
                key_values=key_values,
                n=remaining,
            )
            try:
                result = self._service.acquire_or_generate(
                    query,
                    consumer_id=self._consumer_id,
                    generator_fn=generator_fn,
                )
            except PoolTopupError as exc:
                raise BackendGenerationError(
                    f"failed to generate samples for session {session_id!r}"
                ) from exc
            newly_generated = generated - generated_before
            cache_count = result.claimed - newly_generated
            claimed_from_cache += cache_count

            for idx, sample in enumerate(result.samples):
                source = "pool_cache" if idx < cache_count else "generated"
                responses.append(
                    pool_sample_to_backend_response(
                        sample,
                        source=source,
                        fingerprint=fingerprint,
                    )
                )

            if result.deficit(remaining) > 0:
                time.sleep(0.05)

        return AcquireResult(
            responses=responses[:n],
            claimed_from_cache=claimed_from_cache,
            generated=generated,
        )

    async def acomplete(self, request: BackendRequest) -> BackendResponse:
        return await run_in_thread(lambda: self.complete(request))

    async def aacquire(
        self,
        request: BackendRequest,
        session_id: str,
        n: int,
        *,
        request_id: str | None = None,
    ) -> AcquireResult:
        return await run_in_thread(
            lambda: self.acquire(
                request,
                session_id,
                n,
                request_id=request_id,
            )
        )

    async def adrain(self, timeout: float | None = None) -> DrainResult:
        return await run_in_thread(lambda: self.await_drain(timeout))

    def submit_batch(self, requests: list[BackendRequest]) -> SubmitResult:
        seeded = 0
        skipped = 0
        for request in requests:
            validate_v1_request(request)
            fingerprint = fingerprint_request(request)
            if (
                self._store.complete_count(
                    key_filter=PoolKeyFilter.eq(
                        **{BACKENDS_KEY_COLUMN: fingerprint}
                    )
                )
                > 0
            ):
                skipped += 1
                continue
            sample = PoolSample(
                key_values={BACKENDS_KEY_COLUMN: fingerprint},
                request=backend_request_payload(request),
                response=None,
            )
            if self._store.insert_sample(sample):
                seeded += 1
        return SubmitResult(seeded=seeded, skipped=skipped)

    def await_drain(self, timeout: float | None = None) -> DrainResult:
        controller = start_workers(
            LlmPoolBackend(
                self._store,
                config=LlmPoolBackendConfig(max_retries=1),
            ),
            process_fn=make_backend_process_fn(self._registry),
            config=WorkerConfig(
                num_workers=self._config.num_workers,
                lease_seconds=self._config.lease_seconds,
            ),
        )
        snapshot = self._drain_with_timeout(controller, timeout=timeout)
        backend_state = snapshot.backend_state
        incomplete = 0
        complete = 0
        if backend_state is not None:
            incomplete = int(backend_state.incomplete)
            complete = int(backend_state.complete)
        return DrainResult(
            incomplete=incomplete,
            complete=complete,
            worker_counts=snapshot.counts,
        )

    def _drain_with_timeout(
        self,
        controller: WorkerController[Any],
        *,
        timeout: float | None,
    ) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        poll_interval_s = 0.5
        while True:
            snapshot = controller.snapshot()
            if pool_is_idle(snapshot):
                controller.stop()
                return controller.join()
            if deadline is not None and time.monotonic() >= deadline:
                controller.stop()
                joined = controller.join()
                if not pool_is_idle(joined):
                    raise BackendDrainTimeoutError(
                        "pool drain did not finish before timeout"
                    )
                return joined
            time.sleep(poll_interval_s)

    def _validate_existing_schema(self) -> None:
        existing = catalog.load_schema(self._runtime, self._config.pool_name)
        if existing is None:
            return
        expected = [BACKENDS_KEY_COLUMN]
        if existing.key_column_names != expected:
            raise BackendSchemaError(
                f"pool {self._config.pool_name!r} uses key columns "
                f"{existing.key_column_names!r}; expected {expected!r}"
            )

    def _first_complete_sample(self, fingerprint: str) -> PoolSample | None:
        samples = self._store.bulk_load(
            key_filter=PoolKeyFilter.eq(**{BACKENDS_KEY_COLUMN: fingerprint}),
            completion="complete",
        )
        if not samples:
            return None
        return min(samples, key=lambda sample: sample.sample_idx or 0)

    def _backend_response_to_pool_sample(
        self,
        response: BackendResponse,
        request: BackendRequest,
        key_values: dict[str, Any],
    ) -> PoolSample:
        return PoolSample(
            key_values=key_values,
            request=backend_request_payload(request),
            response=response.model_dump(
                mode="json",
                exclude={"source", "sample_id", "request_fingerprint"},
            ),
            finish_reason=response.finish_reason,
        )

    def _generate_completed_samples(
        self,
        request: BackendRequest,
        key_values: dict[str, Any],
        count: int,
    ) -> list[PoolSample]:
        samples: list[PoolSample] = []
        for _ in range(count):
            response = self._direct.complete(request)
            samples.append(
                self._backend_response_to_pool_sample(
                    response,
                    request,
                    key_values,
                )
            )
        return samples
