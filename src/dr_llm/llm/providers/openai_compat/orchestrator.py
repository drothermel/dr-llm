from __future__ import annotations

from dr_llm.llm.names import ControlStrategy, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.openai_compat.glm_capabilities import (
    reasoning_capabilities_for_glm,
)
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.providers.openai_compat.reasoning import (
    validate_reasoning_for_glm,
    validate_reasoning_for_openai,
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    reasoning_capabilities_for_openai,
)
from dr_llm.llm.providers.openrouter.policy import (
    reasoning_capabilities_for_openrouter,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse


class OpenAICompatOrchestrator:
    def __init__(self, provider: OpenAICompatProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> ProviderName:
        return ProviderName(self._provider.name)

    def model_capabilities(self, model: str) -> ModelCapabilities:
        reasoning = self._resolve_reasoning(model)
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
        provider_name = self._provider.name
        if provider_name == ProviderName.OPENROUTER:
            validate_reasoning_for_openrouter(
                model=request.model, reasoning=request.reasoning
            )
        elif provider_name == ProviderName.GLM:
            validate_reasoning_for_glm(
                model=request.model, reasoning=request.reasoning
            )
        else:
            validate_reasoning_for_openai(
                model=request.model, reasoning=request.reasoning
            )
        return []

    def generate(self, request: LlmRequest) -> LlmResponse:
        return self._provider.generate(request)

    def is_available(self) -> bool:
        return self._provider.availability_status().available

    def close(self) -> None:
        self._provider.close()

    def _resolve_reasoning(self, model: str) -> ReasoningCapabilities | None:
        provider_name = self._provider.name
        if provider_name == ProviderName.OPENROUTER:
            return reasoning_capabilities_for_openrouter(model)
        if provider_name == ProviderName.GLM:
            return reasoning_capabilities_for_glm(model)
        return reasoning_capabilities_for_openai(model)
