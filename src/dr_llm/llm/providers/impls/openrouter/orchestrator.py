from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
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
from dr_llm.llm.providers.impls.openai_compat_base import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.reasoning import (
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.impls.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    apply_openrouter_model_policies,
    openrouter_allowed_models,
    openrouter_model_policy,
    reasoning_capabilities_for_openrouter,
)
from dr_llm.llm.providers.impls.openrouter.provider import OpenRouterProvider
from dr_llm.llm.providers.transports.openai_compat.provider import (
    OpenAICompatProvider,
)
from dr_llm.llm.request import LlmRequest


class OpenRouterOrchestrator(BaseOpenAICompatOrchestrator):
    def __init__(self, provider: OpenAICompatProvider | None = None) -> None:
        super().__init__(provider or OpenRouterProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENROUTER

    def reasoning_capabilities(
        self, model: str
    ) -> ReasoningCapabilities | None:
        return reasoning_capabilities_for_openrouter(model)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_openrouter(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def fetch_models(self):
        entries, raw_payload = super().fetch_models()
        return apply_openrouter_model_policies(entries), raw_payload

    def fallback_models(self):
        models = [
            (model_id, model_id) for model_id in openrouter_allowed_models()
        ]
        entries, raw_payload = build_static_catalog_entries(
            provider=self._provider,
            models=models,
            docs_url="https://openrouter.ai/models",
            supports_vision=None,
            capabilities_fn=self.reasoning_capabilities,
        )
        return apply_openrouter_model_policies(entries), raw_payload

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
