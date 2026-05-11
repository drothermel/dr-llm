from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    CODEX_DOCS_URL,
    CODEX_MODELS,
    build_static_catalog_entries,
)
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
from dr_llm.llm.providers.codex.capabilities import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
    reasoning_capabilities_for_codex,
)
from dr_llm.llm.providers.codex.provider import CodexProvider
from dr_llm.llm.providers.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.orchestrator_base import BaseProviderOrchestrator
from dr_llm.llm.request import LlmRequest


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
        super().validate_request(request)
        validate_reasoning_for_codex(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=CODEX_MODELS,
            docs_url=CODEX_DOCS_URL,
            supports_vision=None,
            capabilities_fn=reasoning_capabilities_for_codex,
        )

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
