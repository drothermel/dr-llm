from __future__ import annotations

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.impls.anthropic.capabilities import (
    anthropic_supports_adaptive_thinking,
    reasoning_capabilities_for_anthropic,
)
from dr_llm.llm.providers.impls.anthropic.effort import (
    supported_effort_levels_for_anthropic,
)
from dr_llm.llm.providers.impls.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.impls.anthropic.reasoning import (
    validate_reasoning_for_anthropic,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest

_ANTHROPIC_COMMON_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]
_ANTHROPIC_DOCS_URL = "https://docs.anthropic.com/en/docs/about-claude/models"


class AnthropicOrchestrator(BaseProviderOrchestrator):
    _provider: AnthropicProvider

    def __init__(self, provider: AnthropicProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.ANTHROPIC

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_anthropic(model),
            supported_effort_levels=supported_effort_levels_for_anthropic(
                model
            ),
        )

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        reasoning = capabilities.reasoning
        if reasoning.mode == ReasoningMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if reasoning.mode == ReasoningMode.ANTHROPIC_BUDGET:
            return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
        if reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT:
            return self._supported_effort_thinking_levels(model)
        if reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET:
            return (
                *self._supported_effort_thinking_levels(model),
                ThinkingLevel.BUDGET,
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
            return AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=self._require_budget_tokens(budget_tokens),
            )
        return AnthropicReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        self._validate_max_tokens_required(request)
        validate_reasoning_for_anthropic(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        defaults = super().request_defaults(model)
        return defaults.model_copy(
            update={
                "max_tokens": 4096,
                "max_tokens_required": True,
                "sampling_supported": True,
                "sampling": SamplingControls(temperature=1.0, top_p=0.95),
            }
        )

    def fetch_models(self):
        return fetch_anthropic_models(
            self._provider,
            capabilities_fn=reasoning_capabilities_for_anthropic,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_ANTHROPIC_COMMON_MODELS,
            docs_url=_ANTHROPIC_DOCS_URL,
            supports_vision=True,
            capabilities_fn=reasoning_capabilities_for_anthropic,
        )

    def _supported_effort_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if anthropic_supports_adaptive_thinking(model):
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        return (ThinkingLevel.OFF,)
