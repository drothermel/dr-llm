from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode, Message
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import ProviderConfig
from dr_llm.providers.usage import TokenUsage

os.environ.setdefault(
    "DR_LLM_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/dr_llm_test",
)


class FakeAdapter(ProviderAdapter):
    def __init__(
        self,
        name: str = "fake",
        config: ProviderConfig | None = None,
        generate_fn: Callable[[LlmRequest], LlmResponse] | None = None,
    ) -> None:
        self._config = config or ProviderConfig(name=name)
        self._generate_fn = generate_fn
        self.close_calls = 0

    def generate(self, request: LlmRequest) -> LlmResponse:
        if self._generate_fn is not None:
            return self._generate_fn(request)
        return make_response(provider=request.provider, model=request.model)

    def close(self) -> None:
        self.close_calls += 1


def make_request(**overrides: Any) -> LlmRequest:
    defaults: dict[str, Any] = {
        "provider": "fake",
        "model": "fake-model",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return LlmRequest(**defaults)


def make_response(**overrides: Any) -> LlmResponse:
    defaults: dict[str, Any] = {
        "text": "response text",
        "provider": "fake",
        "model": "fake-model",
        "mode": CallMode.api,
        "usage": TokenUsage(),
    }
    defaults.update(overrides)
    return LlmResponse(**defaults)


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()
