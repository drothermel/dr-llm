from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import (
    ProviderName,
    ThinkingLevel,
)
from dr_llm.llm.providers.impls.anthropic.capabilities import (
    anthropic_supports_adaptive_thinking,
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
from dr_llm.llm.providers.impls.claude_code.capabilities import (
    reasoning_capabilities_for_claude_code,
    supported_effort_levels_for_claude_code,
)
from dr_llm.llm.providers.impls.claude_code.provider import (
    ClaudeCodeProvider,
)
from dr_llm.llm.providers.impls.claude_code.reasoning import (
    validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.request import LlmRequest

_CLAUDE_CODE_DOCS_URL = "https://code.claude.com/docs/en/model-config"
_CLAUDE_CODE_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]


class ClaudeCodeOrchestrator(BaseProviderOrchestrator):
    _provider: ClaudeCodeProvider

    def __init__(self, provider: ClaudeCodeProvider | None = None) -> None:
        super().__init__(provider or ClaudeCodeProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.CLAUDE_CODE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_claude_code(model),
            supported_effort_levels=supported_effort_levels_for_claude_code(
                model
            ),
        )

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_claude_code(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_CLAUDE_CODE_MODELS,
            docs_url=_CLAUDE_CODE_DOCS_URL,
            supports_vision=True,
            capabilities_fn=reasoning_capabilities_for_claude_code,
        )

    def fallback_models(self):
        return self.fetch_models()

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        del capabilities
        if anthropic_supports_adaptive_thinking(model):
            return (ThinkingLevel.ADAPTIVE,)
        return (ThinkingLevel.NA,)

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if thinking_level == ThinkingLevel.ADAPTIVE:
            return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(
            f"unsupported {self.name} thinking level for model={model!r}: "
            f"{thinking_level!r}"
        )
