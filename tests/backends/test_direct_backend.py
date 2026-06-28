from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dr_llm.backends.direct import DirectBackend
from dr_llm.backends.errors import BackendUnsupportedFeatureError
from dr_llm.llm import CallMode, LlmResponse, ProviderName, TokenUsage
from tests.backends._helpers import make_backend_request


def _response() -> LlmResponse:
    return LlmResponse(
        text="ok",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )


def _registry(response: LlmResponse | None = None) -> MagicMock:
    llm_response = response or _response()
    controls = MagicMock()
    controls.provider = ProviderName.OPENAI
    controls.model = "gpt-4.1-mini"
    controls.mode = CallMode.api
    controls.control_mode = "reasoning"
    controls.supported_thinking_levels = ()
    controls.default_thinking_level = "off"
    controls.supported_effort_levels = ()
    controls.default_effort = "na"
    controls.default_reasoning = None
    defaults = MagicMock()
    defaults.model_dump.return_value = {}
    controls.request_defaults.return_value = defaults
    controls.catalog_metadata = {}

    orchestrator = MagicMock()
    orchestrator.generate.return_value = llm_response
    orchestrator.controls.return_value = controls

    registry = MagicMock()
    registry.get.return_value = orchestrator
    return registry


def test_direct_backend_complete_returns_direct_source() -> None:
    backend = DirectBackend(registry=_registry())
    response = backend.complete(make_backend_request())
    assert response.text == "ok"
    assert response.source == "direct"
    assert response.request_fingerprint is not None


def test_direct_backend_rejects_unsupported_extensions() -> None:
    backend = DirectBackend(registry=_registry())
    with pytest.raises(BackendUnsupportedFeatureError):
        backend.complete(make_backend_request(extensions={"tools": []}))


def test_direct_backend_capabilities_from_controls() -> None:
    backend = DirectBackend(registry=_registry())
    capabilities = backend.capabilities(make_backend_request())
    assert capabilities.provider == ProviderName.OPENAI
    assert capabilities.model == "gpt-4.1-mini"
    assert capabilities.control_mode == "reasoning"
