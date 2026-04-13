from __future__ import annotations

from pydantic import Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.headless.codex_thinking import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
)
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    CodexReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
    dispatch_reasoning_validation,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_budget_range,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    reasoning_capabilities_for_model,
)


def validate_reasoning_for_codex(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: CodexReasoning) -> None:
        if not codex_supports_configurable_thinking(model):
            raise ValueError(f"codex thinking is not supported for model={model!r}")
        validate_discrete_thinking_level(
            provider="codex",
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=codex_supports_off_thinking(model),
            supports_minimal=codex_supports_minimal_thinking(model),
            supports_xhigh=True,
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        capabilities = reasoning_capabilities_for_model(provider="codex", model=model)
        if is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"Reasoning is not supported for provider='codex' model={model!r}"
            )
        assert capabilities is not None
        validate_budget_range(
            provider="codex",
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            capabilities=capabilities,
        )

    dispatch_reasoning_validation(
        provider="codex",
        model=model,
        reasoning=reasoning,
        native_spec_type=CodexReasoning,
        requires_reasoning=codex_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


def validate_reasoning_for_claude_code(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
            raise ValueError(
                f"reasoning is required for provider='claude-code' model={model!r}"
            )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise TypeError(
            f"claude-code does not support budget thinking for model={model!r}"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise TypeError(
            f"claude-code reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError("claude-code does not support anthropic display controls")
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        if reasoning.thinking_level != ThinkingLevel.ADAPTIVE:
            raise ValueError(
                f"claude-code model {model!r} only supports anthropic thinking_level='adaptive'"
            )
        return
    if reasoning.thinking_level != ThinkingLevel.NA:
        raise ValueError(
            f"claude-code model {model!r} does not support explicit anthropic thinking; use thinking_level='na'"
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
            case AnthropicReasoning(thinking_level=ThinkingLevel.NA, display=None):
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


class CodexHeadlessReasoningConfig(BaseProviderReasoningConfig):
    cli_args: list[str] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> CodexHeadlessReasoningConfig:
        if config is None:
            return cls()
        match config:
            case CodexReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case CodexReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(cli_args=["-c", 'model_reasoning_effort="none"'])
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
                | ThinkingLevel.XHIGH
            ):
                thinking_level = config.thinking_level
                return cls(
                    cli_args=["-c", f'model_reasoning_effort="{thinking_level}"']
                )
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("codex headless", config)
        )
