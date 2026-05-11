from __future__ import annotations

from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    GoogleReasoning,
    ReasoningSpec,
    ReasoningWarning,
    google_literal_to_thinking_level,
)
from dr_llm.llm.providers.impls.google.capabilities import (
    reasoning_capabilities_for_google,
)
from dr_llm.llm.providers.impls.google.provider import (
    GoogleProvider,
    GoogleUrls,
)
from dr_llm.llm.providers.impls.google.reasoning import (
    validate_reasoning_for_google,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest

_GOOGLE_COMMON_MODELS = [
    ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash-preview-04-17", "Gemini 2.5 Flash"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
]


class GoogleOrchestrator(BaseProviderOrchestrator):
    _provider: GoogleProvider

    def __init__(self, provider: GoogleProvider | None = None) -> None:
        super().__init__(provider or GoogleProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.GOOGLE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_google(model)
        )

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        reasoning = capabilities.reasoning
        if reasoning.mode == ReasoningMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if reasoning.mode == ReasoningMode.GOOGLE_BUDGET:
            return (
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.OFF,
                ThinkingLevel.BUDGET,
            )
        if reasoning.mode == ReasoningMode.GOOGLE_LEVEL:
            return tuple(
                google_literal_to_thinking_level(level)
                for level in reasoning.google_thinking_levels
            )
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
        del model
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=self._require_budget_tokens(budget_tokens),
            )
        return GoogleReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_google(
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
        return fetch_google_models(
            self._provider,
            capabilities_fn=reasoning_capabilities_for_google,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_GOOGLE_COMMON_MODELS,
            docs_url=GoogleUrls.MODELS_DOCS,
            supports_vision=None,
            capabilities_fn=reasoning_capabilities_for_google,
        )
