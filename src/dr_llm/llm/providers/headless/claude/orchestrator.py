from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import fetch_static_headless_models
from dr_llm.llm.names import (
    ControlStrategy,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.headless.claude.capabilities import (
    reasoning_capabilities_for_claude_code,
    supported_effort_levels_for_claude_code,
)
from dr_llm.llm.providers.headless.claude.provider import (
    ClaudeHeadlessProvider,
)
from dr_llm.llm.providers.headless.claude.reasoning import (
    validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class ClaudeHeadlessOrchestrator(BaseProviderOrchestrator):
    _provider: ClaudeHeadlessProvider

    def __init__(self, provider: ClaudeHeadlessProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.CLAUDE_CODE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_claude_code(model)
        effort_levels = supported_effort_levels_for_claude_code(model)
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

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        super().validate_request(request)
        validate_reasoning_for_claude_code(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return fetch_static_headless_models(self._provider)

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
            return (ThinkingLevel.ADAPTIVE,)
        return (ThinkingLevel.NA,)

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if thinking_level == ThinkingLevel.ADAPTIVE:
            return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(
            f"unsupported {self.name} thinking level for model={model!r}: "
            f"{thinking_level!r}"
        )
