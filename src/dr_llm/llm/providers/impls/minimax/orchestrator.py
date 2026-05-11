from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.config import SamplingControls
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
from dr_llm.llm.providers.impls.minimax.controls import (
    reasoning_capabilities_for_minimax,
    supported_effort_levels_for_minimax,
)
from dr_llm.llm.providers.impls.minimax.families import (
    MiniMaxStaticCatalogModel,
)
from dr_llm.llm.providers.impls.minimax.provider import (
    MiniMaxProvider,
    MiniMaxUrls,
)
from dr_llm.llm.providers.impls.minimax.controls import (
    validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest


class MiniMaxOrchestrator(BaseProviderOrchestrator):
    _provider: MiniMaxProvider

    def __init__(self, provider: MiniMaxProvider | None = None) -> None:
        super().__init__(provider or MiniMaxProvider())

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
        raise ValueError(
            f"{self.name} does not support thinking_level={thinking_level!r}"
        )

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_minimax(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        defaults = super().request_defaults(model)
        return defaults.model_copy(
            update={
                "sampling_supported": True,
                "sampling": SamplingControls(temperature=1.0, top_p=0.95),
            }
        )

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=MiniMaxStaticCatalogModel.values(),
            docs_url=MiniMaxUrls.MODELS_DOCS,
            supports_vision=None,
            capabilities_fn=reasoning_capabilities_for_minimax,
        )

    def fallback_models(self):
        return self.fetch_models()
