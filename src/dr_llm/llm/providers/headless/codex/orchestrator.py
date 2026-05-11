from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import fetch_static_headless_models
from dr_llm.llm.names import (
    ControlStrategy,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    CodexReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.headless.codex.capabilities import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
    reasoning_capabilities_for_codex,
)
from dr_llm.llm.providers.headless.codex.provider import CodexHeadlessProvider
from dr_llm.llm.providers.headless.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class CodexHeadlessOrchestrator(BaseProviderOrchestrator):
    _provider: CodexHeadlessProvider

    def __init__(self, provider: CodexHeadlessProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.CODEX

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_codex(model)
        if reasoning is None:
            reasoning = ReasoningCapabilities(mode=ReasoningMode.UNSUPPORTED)
        control_strategy = (
            ControlStrategy.REASONING
            if reasoning.mode != ReasoningMode.UNSUPPORTED
            else ControlStrategy.NONE
        )
        return ModelCapabilities(
            control_strategy=control_strategy,
            reasoning=reasoning,
            supported_effort_levels=(),
        )

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        super().validate_request(request)
        validate_reasoning_for_codex(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return fetch_static_headless_models(self._provider)

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if not codex_supports_configurable_thinking(model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if codex_supports_off_thinking(model):
            levels.append(ThinkingLevel.OFF)
        elif codex_supports_minimal_thinking(model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [
                ThinkingLevel.LOW,
                ThinkingLevel.MEDIUM,
                ThinkingLevel.HIGH,
                ThinkingLevel.XHIGH,
            ]
        )
        return tuple(levels)

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del model, budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return None
        return CodexReasoning(thinking_level=thinking_level)
