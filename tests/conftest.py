from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from dr_llm.llm import (
    CallMode,
    EffortSpec,
    LlmConfig,
    LlmRequest,
    LlmResponse,
    Message,
    ProviderConfig,
    ProviderRequestDefaults,
    SamplingControls,
    TokenUsage,
    parse_llm_request,
    ProviderName,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.names import ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.core.base import ProviderTransport
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


class FakeProvider(ProviderTransport):
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
    def mode(self) -> CallMode:
        return self._config.mode

    def availability_status(self):
        return self._config.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    def model_capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities(
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

    def build_request(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        effort: EffortSpec | None = None,
        reasoning: ReasoningSpec | None = None,
        thinking_level: ThinkingLevel | None = None,
        budget_tokens: int | None = None,
        sampling: SamplingControls | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        del thinking_level, budget_tokens
        payload: dict[str, Any] = {
            "provider": self.name,
            "model": model,
            "mode": self.mode,
            "messages": messages,
            "effort": EffortSpec.NA if effort is None else effort,
            "reasoning": reasoning,
            "sampling": sampling,
            "metadata": metadata or {},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return parse_llm_request(payload)

    def build_config(
        self,
        *,
        model: str,
        max_tokens: int | None = None,
        effort: EffortSpec | None = None,
        reasoning: ReasoningSpec | None = None,
        thinking_level: ThinkingLevel | None = None,
        budget_tokens: int | None = None,
        sampling: SamplingControls | None = None,
    ) -> LlmConfig:
        del thinking_level, budget_tokens
        payload: dict[str, Any] = {
            "provider": self.name,
            "model": model,
            "mode": self.mode,
            "effort": EffortSpec.NA if effort is None else effort,
            "reasoning": reasoning,
            "sampling": sampling,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return LlmConfig(**payload)

    def build_request_from_config(
        self,
        *,
        config: LlmConfig,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        return parse_llm_request(
            {
                **config.model_dump(mode="python"),
                "messages": messages,
                "metadata": metadata or {},
            }
        )

    def validate_config(self, config: LlmConfig) -> None:
        if config.provider != self.name:
            raise ValueError(
                f"config provider {config.provider!r} does not match "
                f"orchestrator provider {self.name!r}"
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
        "mode": CallMode.api,
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    if "mode" not in overrides and defaults["provider"] in {
        ProviderName.CODEX,
        ProviderName.CLAUDE_CODE,
        str(ProviderName.CODEX),
        str(ProviderName.CLAUDE_CODE),
    }:
        defaults["mode"] = CallMode.headless
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
