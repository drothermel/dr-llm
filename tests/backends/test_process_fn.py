from __future__ import annotations

from unittest.mock import MagicMock

from dr_llm.backends.process_fn import make_backend_process_fn
from dr_llm.llm import CallMode, LlmResponse, ProviderName, TokenUsage
from dr_llm.pool.pool_sample import PoolSample
from tests.backends._helpers import make_backend_request


def test_make_backend_process_fn_generates_from_sample_request() -> None:
    request = make_backend_request()
    response = LlmResponse(
        text="worker",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )
    orchestrator = MagicMock()
    orchestrator.generate.return_value = response
    registry = MagicMock()
    registry.get.return_value = orchestrator

    process_fn = make_backend_process_fn(registry)
    sample = PoolSample(
        key_values={"request_fingerprint": "fp"},
        request={
            "backend_request": request.model_dump(mode="json"),
        },
    )

    result = process_fn(sample)

    assert result.text == "worker"
    orchestrator.generate.assert_called_once()
