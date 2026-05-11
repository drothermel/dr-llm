from __future__ import annotations

from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.names import ControlStrategy, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.google.capabilities import (
    reasoning_capabilities_for_google,
)
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.google.reasoning import validate_reasoning_for_google
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class GoogleOrchestrator(BaseProviderOrchestrator):
    _provider: GoogleProvider

    def __init__(self, provider: GoogleProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.GOOGLE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = reasoning_capabilities_for_google(model)
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
        validate_reasoning_for_google(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return fetch_google_models(self._provider)
