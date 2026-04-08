"""LLM-backed process function adapter for pool workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from dr_llm.llm.config import LlmConfig
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.registry import ProviderRegistry
from dr_llm.logging.events import generation_log_context, get_generation_log_context
from dr_llm.logging.sinks import emit_generation_event
from dr_llm.pool.pending.pending_sample import PendingSample

ProcessFn = Callable[[PendingSample], dict[str, Any]]


def _require_payload_field(sample: PendingSample, key: str) -> Any:
    value = sample.payload.get(key)
    if value is None:
        raise KeyError(
            f"Pending sample payload missing {key!r}; "
            "was the pool seeded with a rich grid for this column?"
        )
    return value


def make_llm_process_fn(
    registry: ProviderRegistry,
    *,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> ProcessFn:
    """Build a ``ProcessFn`` that dispatches LLM calls via the provider registry.

    Expects pending samples whose ``payload`` contains serialized
    :class:`LlmConfig` (under *llm_config_key*) and ``list[Message]``
    (under *prompt_key*).
    """

    def _process(sample: PendingSample) -> dict[str, Any]:
        raw_config = _require_payload_field(sample, llm_config_key)
        raw_messages = _require_payload_field(sample, prompt_key)

        config = LlmConfig(**raw_config)
        messages = [Message(**message) for message in raw_messages]
        request = config.to_request(messages)

        context = get_generation_log_context()
        model_provider = registry.get(request.provider)
        call_id = uuid4().hex
        worker_payload = {
            "pool_name": context.get("pool_name"),
            "pending_id": sample.pending_id,
            "sample_idx": sample.sample_idx,
            "worker_id": context.get("worker_id"),
            "key_values": sample.key_values,
        }

        with generation_log_context(
            {
                "call_id": call_id,
                "provider": request.provider,
                "model": request.model,
                "mode": model_provider.mode,
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
                response = model_provider.generate(request)
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
            return response.model_dump()

    return _process
