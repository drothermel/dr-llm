from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    MINIMAX_DOCS_URL,
    MINIMAX_TEXT_MODELS,
    build_static_catalog_entries,
)
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.impls.minimax.capabilities import (
    reasoning_capabilities_for_minimax,
    supported_effort_levels_for_minimax,
)
from dr_llm.llm.providers.impls.minimax.provider import MiniMaxProvider
from dr_llm.llm.providers.impls.minimax.reasoning import (
    validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.request import LlmRequest


class MiniMaxOrchestrator(BaseProviderOrchestrator):
    _provider: MiniMaxProvider

    def __init__(self, provider: MiniMaxProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.MINIMAX

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_minimax(model),
            supported_effort_levels=supported_effort_levels_for_minimax(model),
        )

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        reasoning = capabilities.reasoning
        if reasoning.mode in {
            ReasoningMode.UNSUPPORTED,
            ReasoningMode.MINIMAX_EFFORT,
        }:
            return (ThinkingLevel.NA,)
        raise ValueError(
            f"unexpected reasoning mode for provider={self.name!r} "
            f"model={model!r}: {reasoning.mode!r}"
        )

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del model, budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        return AnthropicReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        super().validate_request(request)
        validate_reasoning_for_minimax(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=MINIMAX_TEXT_MODELS,
            docs_url=MINIMAX_DOCS_URL,
            supports_vision=None,
            capabilities_fn=reasoning_capabilities_for_minimax,
        )
