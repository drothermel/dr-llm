from __future__ import annotations

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.names import (
    ControlStrategy,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.anthropic.capabilities import (
    reasoning_capabilities_for_anthropic,
)
from dr_llm.llm.providers.anthropic.effort import (
    supported_effort_levels_for_anthropic,
)
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.anthropic.reasoning import (
    validate_reasoning_for_anthropic,
)
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class AnthropicOrchestrator(BaseProviderOrchestrator):
    _provider: AnthropicProvider

    def __init__(self, provider: AnthropicProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.ANTHROPIC

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_anthropic(model)
        effort_levels = supported_effort_levels_for_anthropic(model)
        if reasoning is None:
            reasoning = ReasoningCapabilities(mode=ReasoningMode.UNSUPPORTED)
        control_strategy = (
            ControlStrategy.EFFORT
            if effort_levels
            else (
                ControlStrategy.REASONING
                if reasoning.mode != ReasoningMode.UNSUPPORTED
                else ControlStrategy.NONE
            )
        )
        return ModelCapabilities(
            control_strategy=control_strategy,
            reasoning=reasoning,
            supported_effort_levels=effort_levels,
        )

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        capabilities = self.model_capabilities(model).reasoning
        if capabilities.mode == ReasoningMode.ANTHROPIC_EFFORT:
            return self._supported_effort_thinking_levels(model)
        if capabilities.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET:
            return (
                *self._supported_effort_thinking_levels(model),
                ThinkingLevel.BUDGET,
            )
        return super().supported_thinking_levels(model)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        super().validate_request(request)
        self._validate_max_tokens_required(request)
        validate_reasoning_for_anthropic(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return fetch_anthropic_models(self._provider)

    def _supported_effort_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        return (ThinkingLevel.OFF,)
