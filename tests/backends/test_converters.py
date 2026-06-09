from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dr_llm.backends.converters import (
    backend_request_from_sample,
    capabilities_from_controls,
)
from dr_llm.backends.errors import BackendValidationError
from dr_llm.backends.models import BackendRequest
from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.pool.pool_sample import PoolSample


def test_backend_request_round_trip_to_llm_request() -> None:
    backend_request = BackendRequest(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        messages=[Message(role="user", content="hello")],
        metadata={"trace": "1"},
        extensions={"note": "ignored-by-llm"},
    )
    llm_request = backend_request.to_llm_request()
    assert isinstance(llm_request, LlmRequest)
    assert llm_request.model == "gpt-4.1-mini"
    assert llm_request.metadata == {"trace": "1"}

    round_trip = BackendRequest.from_llm_request(
        llm_request,
        extensions=backend_request.extensions,
    )
    assert round_trip == backend_request


def test_capabilities_from_controls_preserves_none_defaults() -> None:
    controls = MagicMock()
    controls.provider = ProviderName.OPENAI
    controls.model = "gpt-4.1-mini"
    controls.mode = CallMode.api
    controls.control_mode = "reasoning"
    controls.supported_thinking_levels = ()
    controls.default_thinking_level = None
    controls.supported_effort_levels = ()
    controls.default_effort = None
    controls.default_reasoning = None
    defaults = MagicMock()
    defaults.model_dump.return_value = {}
    controls.request_defaults.return_value = defaults
    controls.catalog_metadata = {}

    capabilities = capabilities_from_controls(controls)

    assert capabilities.default_thinking_level is None
    assert capabilities.default_effort is None


def test_backend_request_from_sample_raises_backend_validation_error() -> None:
    sample = PoolSample(
        key_values={"request_fingerprint": "fp"},
        request={"backend_request": {"provider": "not-a-provider"}},
    )

    with pytest.raises(BackendValidationError, match="invalid request"):
        backend_request_from_sample(sample)
