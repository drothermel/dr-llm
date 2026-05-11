from __future__ import annotations

from dr_llm.llm import ProviderName
import os
from collections.abc import Callable
from typing import Any

import pytest

from dr_llm.llm import (
    CallMode,
    EffortSpec,
    LlmRequest,
    LlmResponse,
    Message,
    ProviderConfig,
    ProviderRequestDefaults,
    TokenUsage,
    parse_llm_request,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.names import ControlStrategy, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.core.base import Provider
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.reasoning_controls import ReasoningControls

os.environ.setdefault(
    "DR_LLM_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/dr_llm_test",
)


class FakeProvider(Provider):
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


class FakeOrchestrator:
    def __init__(
        self,
        name: str = "fake",
        config: ProviderConfig | None = None,
        generate_fn: Callable[[LlmRequest], LlmResponse] | None = None,
        fetch_models_fn: Callable[
            [], tuple[list[ModelCatalogEntry], dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        self._config = config or ProviderConfig(name=name)
        self._generate_fn = generate_fn
        self._fetch_models_fn = fetch_models_fn
        self.close_calls = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def mode(self) -> str:
        return self._config.mode

    def availability_status(self):
        return self._config.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    def model_capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities(
            control_strategy=ControlStrategy.NONE,
            reasoning=ReasoningCapabilities(mode=ReasoningMode.UNSUPPORTED),
        )

    def reasoning_controls(self, model: str) -> ReasoningControls:
        return ReasoningControls(
            provider=self.name,
            model=model,
            supported_thinking_levels=(ThinkingLevel.NA,),
            default_thinking_level=ThinkingLevel.NA,
            supported_effort_levels=(),
            default_effort=EffortSpec.NA,
            default_reasoning=None,
        )

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        return ProviderRequestDefaults(
            provider=self.name,
            model=model,
            mode=self.mode,
            effort=EffortSpec.NA,
            reasoning=None,
        )

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del model, thinking_level, budget_tokens
        return None

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        del request
        return []

    def generate(self, request: LlmRequest) -> LlmResponse:
        if self._generate_fn is not None:
            return self._generate_fn(request)
        return make_response(provider=request.provider, model=request.model)

    def fetch_models(self) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        if self._fetch_models_fn is not None:
            return self._fetch_models_fn()
        return [], {"source": "fake"}

    def fallback_models(
        self,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return self.fetch_models()

    def close(self) -> None:
        self.close_calls += 1


def make_request(**overrides: Any) -> LlmRequest:
    defaults: dict[str, Any] = {
        "provider": ProviderName.OPENAI,
        "model": "gpt-4.1-mini",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return parse_llm_request(defaults)


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
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def fake_orchestrator() -> FakeOrchestrator:
    return FakeOrchestrator()
