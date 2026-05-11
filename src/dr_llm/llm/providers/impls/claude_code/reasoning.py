from __future__ import annotations

from pydantic import Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.impls.anthropic.capabilities import (
    anthropic_supports_adaptive_thinking,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    unsupported_reasoning_kind_message,
)


def validate_reasoning_for_claude_code(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        if anthropic_supports_adaptive_thinking(model):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.CLAUDE_CODE}' model={model!r}"
            )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise TypeError(
            f"{ProviderName.CLAUDE_CODE} does not support budget thinking for model={model!r}"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise TypeError(
            f"{ProviderName.CLAUDE_CODE} reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError(
            f"{ProviderName.CLAUDE_CODE} does not support anthropic display controls"
        )
    if anthropic_supports_adaptive_thinking(model):
        if reasoning.thinking_level != ThinkingLevel.ADAPTIVE:
            raise ValueError(
                f"{ProviderName.CLAUDE_CODE} model {model!r} only supports anthropic thinking_level='adaptive'"
            )
        return
    if reasoning.thinking_level != ThinkingLevel.NA:
        raise ValueError(
            f"{ProviderName.CLAUDE_CODE} model {model!r} does not support explicit anthropic thinking; use thinking_level='na'"
        )


class ClaudeHeadlessReasoningConfig(BaseProviderReasoningConfig):
    cli_args: list[str] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> ClaudeHeadlessReasoningConfig:
        if config is None:
            return cls()
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA, display=None
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls()
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("claude headless", config)
        )
