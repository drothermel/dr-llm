from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import (
    LlmResponse,
    Message,
    ProviderRegistry,
    parse_llm_config,
)
from dr_llm.logging.events import (
    generation_log_context,
    get_generation_log_context,
)
from dr_llm.logging.sinks import emit_generation_event
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.db.sql_helpers import validate_key_filter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.workers import ErrorDecision, ProcessFn, WorkerBackend

logger = logging.getLogger(__name__)


class LlmPoolBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(default=0, ge=0)
    key_filter: PoolKeyFilter | None = None


class LlmPoolBackendState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incomplete: int = 0
    complete: int = 0
    key_filter: PoolKeyFilter | None = None
    max_retries: int = 0


LlmWorkerBackend = WorkerBackend[PoolSample, LlmResponse, LlmPoolBackendState]
LlmProcessFn = ProcessFn[PoolSample, LlmResponse]


class LlmPoolBackend(
    WorkerBackend[PoolSample, LlmResponse, LlmPoolBackendState]
):
    """LLM-specific backend for the generic worker runtime."""

    def __init__(
        self, store: PoolStore, *, config: LlmPoolBackendConfig
    ) -> None:
        self._store = store
        self._config = config
        if config.key_filter is not None:
            validate_key_filter(self._store.schema, config.key_filter)

    def claim(
        self, *, worker_id: str, lease_seconds: int
    ) -> PoolSample | None:
        sample = self._store.claim_lease(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            key_filter=self._config.key_filter,
        )
        return sample

    def complete(
        self,
        *,
        item: PoolSample,
        result: LlmResponse,
        worker_id: str,
    ) -> None:
        attempt_count = self._attempt_count(item)
        completed = self._store.complete_sample(
            sample_id=item.sample_id,
            response=result.model_dump(),
            finish_reason=result.finish_reason,
            attempt_count=attempt_count,
            lease_owner=worker_id,
        )
        released = self._store.release_lease(
            sample_id=item.sample_id,
            worker_id=worker_id,
        )
        if not completed:
            raise RuntimeError(
                f"Failed to complete leased pool sample {item.sample_id}"
            )
        if not released:
            logger.warning(
                "release_lease no-op: sample_id=%s worker_id=%s "
                "(lease was stale or re-leased)",
                item.sample_id,
                worker_id,
            )

    def handle_process_error(
        self,
        *,
        item: PoolSample,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        attempt_count = self._attempt_count(item)
        if attempt_count <= self._config.max_retries:
            released = self._store.release_lease(
                sample_id=item.sample_id,
                worker_id=worker_id,
            )
            if not released:
                logger.warning(
                    "release_lease no-op: sample_id=%s worker_id=%s "
                    "(lease was stale or re-leased)",
                    item.sample_id,
                    worker_id,
                )
            return ErrorDecision.retry

        message = str(exc).strip()
        reason = (
            f"{type(exc).__name__}: {message}"
            if message
            else type(exc).__name__
        )
        completed = self._store.complete_sample(
            sample_id=item.sample_id,
            response={"error": reason},
            finish_reason="error",
            attempt_count=attempt_count,
            lease_owner=worker_id,
        )
        released = self._store.release_lease(
            sample_id=item.sample_id,
            worker_id=worker_id,
        )
        if not completed:
            logger.warning(
                "complete_sample no-op: sample_id=%s worker_id=%s "
                "(sample was already complete or missing)",
                item.sample_id,
                worker_id,
            )
        if not released:
            logger.warning(
                "release_lease no-op: sample_id=%s worker_id=%s "
                "(lease was stale or re-leased)",
                item.sample_id,
                worker_id,
            )
        return ErrorDecision.fail

    def snapshot(self) -> LlmPoolBackendState:
        return LlmPoolBackendState(
            incomplete=self._store.incomplete_count(
                key_filter=self._config.key_filter
            ),
            complete=self._store.complete_count(
                key_filter=self._config.key_filter
            ),
            key_filter=(
                None
                if self._config.key_filter is None
                else self._config.key_filter.model_copy(deep=True)
            ),
            max_retries=self._config.max_retries,
        )

    def process_context(
        self,
        *,
        item: PoolSample,
        worker_id: str,
    ) -> AbstractContextManager[Any]:
        return generation_log_context(
            {
                "pool_name": self._store.schema.name,
                "worker_id": worker_id,
            }
        )

    def _attempt_count(self, item: PoolSample) -> int:
        return max(item.attempt_count, 1)


# ---------------------------------------------------------------------------
# LLM process function
# ---------------------------------------------------------------------------


def _require_request_field(sample: PoolSample, key: str) -> Any:
    if key not in sample.request:
        raise KeyError(
            f"PoolSample.request is missing key {key!r}; "
            "was the pool seeded with a rich grid for this column?"
        )
    value = sample.request[key]
    if value is None:
        raise ValueError(
            f"PoolSample.request[{key!r}] is present but has explicit None value"
        )
    return value


def make_llm_process_fn(
    registry: ProviderRegistry,
    *,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> LlmProcessFn:
    """Build a process function that dispatches LLM calls via the provider registry.

    Expects pool samples whose ``request`` contains serialized
    :class:`LlmConfig` (under *llm_config_key*) and ``list[Message]``
    (under *prompt_key*).
    """

    def _process(sample: PoolSample) -> LlmResponse:
        raw_config = _require_request_field(sample, llm_config_key)
        raw_messages = _require_request_field(sample, prompt_key)

        config = parse_llm_config(raw_config)
        messages = [Message(**message) for message in raw_messages]

        context = get_generation_log_context()
        orchestrator = registry.get(config.provider)
        request = orchestrator.build_request(
            model=config.model,
            messages=messages,
            max_tokens=getattr(config, "max_tokens", None),
            effort=config.effort,
            reasoning=config.reasoning,
            temperature=getattr(config, "temperature", None),
            top_p=getattr(config, "top_p", None),
        )
        call_id = uuid4().hex
        worker_payload = {
            "pool_name": context.get("pool_name"),
            "sample_id": sample.sample_id,
            "sample_idx": sample.sample_idx,
            "worker_id": context.get("worker_id"),
            "key_values": sample.key_values,
        }

        with generation_log_context(
            {
                "call_id": call_id,
                "provider": request.provider,
                "model": request.model,
                "mode": orchestrator.mode,
            }
        ):
            emit_generation_event(
                event_type="llm_call.started",
                stage="pool_worker.before_provider",
                payload={
                    **worker_payload,
                    "request": request.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    ),
                },
            )
            try:
                response = orchestrator.generate(request)
            except Exception as exc:  # noqa: BLE001
                emit_generation_event(
                    event_type="llm_call.failed",
                    stage="pool_worker.provider_exception",
                    payload={
                        **worker_payload,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            emit_generation_event(
                event_type="llm_call.succeeded",
                stage="pool_worker.after_provider",
                payload={
                    **worker_payload,
                    "response": response.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    ),
                },
            )
            return response

    return _process
