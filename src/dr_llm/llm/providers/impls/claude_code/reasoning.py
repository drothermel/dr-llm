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
            msg = (
                "reasoning is required for "
                f"provider='{ProviderName.CLAUDE_CODE}' model={model!r}"
            )
            raise ValueError(msg)
        return
    if isinstance(reasoning, ReasoningBudget):
        msg = f"{ProviderName.CLAUDE_CODE} does not support budget thinking for model={model!r}"
        raise TypeError(msg)
    if not isinstance(reasoning, AnthropicReasoning):
        msg = f"{ProviderName.CLAUDE_CODE} reasoning is not supported for kind={reasoning.kind!r}"
        raise TypeError(msg)
    if reasoning.display is not None:
        msg = (
            f"{ProviderName.CLAUDE_CODE} does not support anthropic display "
            "controls"
        )
        raise ValueError(msg)
    if reasoning.budget_tokens is not None:
        msg = f"{ProviderName.CLAUDE_CODE} does not support budget_tokens"
        raise ValueError(msg)
    if anthropic_supports_adaptive_thinking(model):
        if reasoning.thinking_level != ThinkingLevel.ADAPTIVE:
            msg = f"{ProviderName.CLAUDE_CODE} model {model!r} only supports anthropic thinking_level='adaptive'"
            raise ValueError(msg)
        return
    if reasoning.thinking_level != ThinkingLevel.NA:
        msg = f"{ProviderName.CLAUDE_CODE} model {model!r} does not support explicit anthropic thinking; use thinking_level='na'"
        raise ValueError(msg)


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
