from __future__ import annotations

from dr_llm.llm.names import ControlStrategy, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.minimax.capabilities import (
    reasoning_capabilities_for_minimax,
    supported_effort_levels_for_minimax,
)
from dr_llm.llm.providers.minimax.provider import MiniMaxProvider
from dr_llm.llm.providers.minimax.reasoning import (
    validate_reasoning_for_minimax,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse


class MiniMaxOrchestrator:
    def __init__(self, provider: MiniMaxProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> ProviderName:
        return ProviderName.MINIMAX

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_minimax(model)
        effort_levels = supported_effort_levels_for_minimax(model)
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
        validate_reasoning_for_minimax(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def generate(self, request: LlmRequest) -> LlmResponse:
        return self._provider.generate(request)

    def is_available(self) -> bool:
        return self._provider.availability_status().available

    def close(self) -> None:
        self._provider.close()
