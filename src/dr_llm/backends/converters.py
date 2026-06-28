"""Convert between backend models and existing dr-llm types."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from dr_llm.backends.errors import BackendValidationError
from dr_llm.backends.models import (
    BackendCapabilities,
    BackendRequest,
    BackendResponse,
    ResponseSource,
)
from dr_llm.llm.providers.core.controls import ProviderControls
from dr_llm.llm.response import LlmResponse
from dr_llm.pool.pool_sample import PoolSample

_LLM_RESPONSE_ADAPTER = TypeAdapter(LlmResponse)


def llm_response_to_backend_response(
    response: LlmResponse,
    *,
    source: ResponseSource | None = None,
    sample_id: str | None = None,
    fingerprint: str | None = None,
) -> BackendResponse:
    payload = response.model_dump()
    return BackendResponse(
        **payload,
        source=source,
        sample_id=sample_id,
        request_fingerprint=fingerprint,
    )


def pool_sample_to_backend_response(
    sample: PoolSample,
    *,
    source: ResponseSource,
    fingerprint: str | None = None,
) -> BackendResponse:
    if sample.response is None:
        msg = f"Pool sample {sample.sample_id} has no response payload"
        raise ValueError(msg)
    llm_response = _LLM_RESPONSE_ADAPTER.validate_python(sample.response)
    return llm_response_to_backend_response(
        llm_response,
        source=source,
        sample_id=sample.sample_id,
        fingerprint=fingerprint,
    )


def capabilities_from_controls(
    controls: ProviderControls,
) -> BackendCapabilities:
    default_reasoning = controls.default_reasoning
    return BackendCapabilities(
        provider=str(controls.provider),
        model=controls.model,
        mode=controls.mode,
        control_mode=str(controls.control_mode),
        supported_thinking_levels=tuple(
            str(level) for level in controls.supported_thinking_levels
        ),
        default_thinking_level=(
            None
            if controls.default_thinking_level is None
            else str(controls.default_thinking_level)
        ),
        supported_effort_levels=tuple(
            str(level) for level in controls.supported_effort_levels
        ),
        default_effort=(
            None
            if controls.default_effort is None
            else str(controls.default_effort)
        ),
        default_reasoning=(
            default_reasoning.model_dump(mode="json")
            if default_reasoning is not None
            else None
        ),
        request_defaults=controls.request_defaults().model_dump(mode="json"),
        catalog_metadata=controls.catalog_metadata,
    )


def backend_request_payload(request: BackendRequest) -> dict[str, Any]:
    return {
        "backend_request": request.model_dump(
            mode="json",
            exclude_none=True,
        )
    }


def backend_request_from_sample(sample: PoolSample) -> BackendRequest:
    raw = sample.request.get("backend_request")
    if raw is None:
        msg = f"Pool sample {sample.sample_id} is missing request['backend_request']"
        raise KeyError(msg)
    try:
        return BackendRequest(**raw)
    except ValidationError as exc:
        msg = f"Pool sample {sample.sample_id} has invalid request['backend_request']"
        raise BackendValidationError(msg) from exc
