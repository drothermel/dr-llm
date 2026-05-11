from __future__ import annotations

from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    OpenRouterReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.transports.openai_compat.orchestrator import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.reasoning import (
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.impls.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
    reasoning_capabilities_for_openrouter,
)
from dr_llm.llm.request import LlmRequest


class OpenRouterOrchestrator(BaseOpenAICompatOrchestrator):
    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENROUTER

    def reasoning_capabilities(
        self, model: str
    ) -> ReasoningCapabilities | None:
        return reasoning_capabilities_for_openrouter(model)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        super().validate_request(request)
        validate_reasoning_for_openrouter(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        del model, capabilities
        return (ThinkingLevel.NA,)

    def default_reasoning(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> ReasoningSpec | None:
        del thinking_level, capabilities
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
        raise ValueError(
            f"openrouter does not support thinking_level={thinking_level!r}"
        )
