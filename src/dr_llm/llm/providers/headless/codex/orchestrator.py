from __future__ import annotations

from dr_llm.llm.names import ControlStrategy, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.headless.codex.capabilities import (
    reasoning_capabilities_for_codex,
)
from dr_llm.llm.providers.headless.codex.provider import CodexHeadlessProvider
from dr_llm.llm.providers.headless.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse


class CodexHeadlessOrchestrator:
    def __init__(self, provider: CodexHeadlessProvider) -> None:
        self._provider = provider

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
        validate_reasoning_for_codex(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def generate(self, request: LlmRequest) -> LlmResponse:
        return self._provider.generate(request)

    def is_available(self) -> bool:
        return self._provider.availability_status().available

    def close(self) -> None:
        self._provider.close()
