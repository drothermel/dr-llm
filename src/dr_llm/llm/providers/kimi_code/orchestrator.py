from __future__ import annotations

from dr_llm.llm.catalog.fetchers.kimi import fetch_kimi_models
from dr_llm.llm.names import ControlStrategy, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.kimi_code.capabilities import (
    reasoning_capabilities_for_kimi_code,
    supported_effort_levels_for_kimi_code,
)
from dr_llm.llm.providers.kimi_code.provider import KimiCodeProvider
from dr_llm.llm.providers.kimi_code.reasoning import (
    validate_reasoning_for_kimi_code,
)
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class KimiCodeOrchestrator(BaseProviderOrchestrator):
    _provider: KimiCodeProvider

    def __init__(self, provider: KimiCodeProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.KIMI_CODE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_kimi_code(model)
        effort_levels = supported_effort_levels_for_kimi_code(model)
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
        self._validate_max_tokens_required(request)
        validate_reasoning_for_kimi_code(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return fetch_kimi_models(self._provider)
