from __future__ import annotations

from dr_llm.llm.catalog.fetchers.openai_compat import (
    fetch_openai_compat_models,
)
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
    GlmReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
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
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
    reasoning_capabilities_for_openai,
    validate_openai_sampling_controls,
)
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
)
from dr_llm.llm.providers.openrouter.policy import (
    openrouter_model_policy,
    reasoning_capabilities_for_openrouter,
)
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


class OpenAICompatOrchestrator(BaseProviderOrchestrator):
    _provider: OpenAICompatProvider

    def __init__(self, provider: OpenAICompatProvider) -> None:
        super().__init__(provider)

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
        super().validate_request(request)
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
            validate_openai_sampling_controls(
                model=request.model,
                reasoning=request.reasoning,
                temperature=getattr(request, "temperature", None),
                top_p=getattr(request, "top_p", None),
            )
        return []

    def fetch_models(self):
        return fetch_openai_compat_models(self._provider)

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        provider_name = self._provider.name
        if provider_name == ProviderName.OPENAI:
            return self._supported_openai_thinking_levels(model)
        if provider_name == ProviderName.OPENROUTER:
            return (ThinkingLevel.NA,)
        return super().supported_thinking_levels(model)

    def default_reasoning(
        self, *, model: str, thinking_level: ThinkingLevel | None = None
    ) -> ReasoningSpec | None:
        if self._provider.name == ProviderName.OPENROUTER:
            return self._default_openrouter_reasoning(model)
        return super().default_reasoning(
            model=model, thinking_level=thinking_level
        )

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        provider_name = self._provider.name
        if provider_name == ProviderName.OPENAI:
            if thinking_level == ThinkingLevel.NA:
                return None
            return OpenAIReasoning(thinking_level=thinking_level)
        if provider_name == ProviderName.GLM:
            if thinking_level == ThinkingLevel.NA:
                return None
            return GlmReasoning(thinking_level=thinking_level)
        if provider_name == ProviderName.OPENROUTER:
            if thinking_level == ThinkingLevel.NA:
                return None
            raise ValueError(
                f"openrouter does not support thinking_level={thinking_level!r}"
            )
        raise ValueError(f"unsupported provider: {provider_name!r}")

    def _resolve_reasoning(self, model: str) -> ReasoningCapabilities | None:
        provider_name = self._provider.name
        if provider_name == ProviderName.OPENROUTER:
            return reasoning_capabilities_for_openrouter(model)
        if provider_name == ProviderName.GLM:
            return reasoning_capabilities_for_glm(model)
        return reasoning_capabilities_for_openai(model)

    def _supported_openai_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if not openai_supports_configurable_thinking(model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if openai_supports_off_thinking(model):
            levels.append(ThinkingLevel.OFF)
        elif openai_supports_minimal_thinking(model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
        )
        return tuple(levels)

    def _default_openrouter_reasoning(
        self, model: str
    ) -> OpenRouterReasoning | None:
        policy = openrouter_model_policy(model)
        if policy is None:
            raise ValueError(f"missing openrouter policy for model={model!r}")
        if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
            return None
        if (
            policy.request_style
            == OpenRouterReasoningRequestStyle.ENABLED_FLAG
        ):
            if policy.supports_disable:
                return OpenRouterReasoning(enabled=False)
            return OpenRouterReasoning(enabled=True)
        return OpenRouterReasoning(effort=policy.allowed_efforts[0])
