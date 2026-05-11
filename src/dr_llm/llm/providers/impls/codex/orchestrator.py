from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import (
    ProviderName,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    CodexReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.impls.codex.capabilities import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
    reasoning_capabilities_for_codex,
)
from dr_llm.llm.providers.impls.codex.provider import CodexProvider
from dr_llm.llm.providers.impls.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.request import LlmRequest

_CODEX_DOCS_URL = "https://developers.openai.com/codex/models"
_CODEX_MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4 Mini"),
    ("gpt-5.3-codex", "GPT-5.3 Codex"),
    ("gpt-5.3-codex-spark", "GPT-5.3 Codex Spark (Pro only)"),
    ("gpt-5.2-codex", "GPT-5.2 Codex"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5-codex", "GPT-5 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5", "GPT-5"),
]


class CodexOrchestrator(BaseProviderOrchestrator):
    _provider: CodexProvider

    def __init__(self, provider: CodexProvider) -> None:
        super().__init__(provider)

    @property
    def name(self) -> ProviderName:
        return ProviderName.CODEX

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_codex(model)
        )

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_codex(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_CODEX_MODELS,
            docs_url=_CODEX_DOCS_URL,
            supports_vision=None,
            capabilities_fn=reasoning_capabilities_for_codex,
        )

    def fallback_models(self):
        return self.fetch_models()

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        del capabilities
        if not codex_supports_configurable_thinking(model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if codex_supports_off_thinking(model):
            levels.append(ThinkingLevel.OFF)
        elif codex_supports_minimal_thinking(model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [
                ThinkingLevel.LOW,
                ThinkingLevel.MEDIUM,
                ThinkingLevel.HIGH,
                ThinkingLevel.XHIGH,
            ]
        )
        return tuple(levels)

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
        return CodexReasoning(thinking_level=thinking_level)
