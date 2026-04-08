from __future__ import annotations

from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
    ANTHROPIC_BUDGET_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
    dispatch_reasoning_validation,
    is_reasoning_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_budget_range,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    reasoning_capabilities_for_model,
)


def _validate_anthropic_budget_for_provider(
    *, provider: str, model: str, budget_tokens: int | None
) -> None:
    if budget_tokens is None:
        raise ValueError(
            f"{provider} budget thinking requires budget_tokens when "
            "thinking_level is 'budget'"
        )
    capabilities = reasoning_capabilities_for_model(provider=provider, model=model)
    if (
        capabilities is None
        or capabilities.min_budget_tokens is None
        or capabilities.max_budget_tokens is None
    ):
        raise ValueError(
            f"{provider} budget thinking is not supported for model={model!r}"
        )
    validate_budget_range(
        provider=provider,
        model=model,
        label=f"{provider} budget_tokens",
        tokens=budget_tokens,
        capabilities=capabilities,
    )


def validate_reasoning_for_anthropic(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    capabilities = reasoning_capabilities_for_model(provider="anthropic", model=model)
    dispatch_reasoning_validation(
        provider="anthropic",
        model=model,
        reasoning=reasoning,
        native_spec_type=AnthropicReasoning,
        requires_reasoning=not is_reasoning_unsupported(capabilities),
        validate_native=lambda spec: _validate_anthropic_reasoning_shape(
            model=model, reasoning=spec
        ),
        validate_top_budget=lambda budget: _validate_anthropic_budget_for_provider(
            provider="anthropic", model=model, budget_tokens=budget.tokens
        ),
    )


def _validate_anthropic_reasoning_shape(
    *, model: str, reasoning: AnthropicReasoning
) -> None:
    thinking_level = reasoning.thinking_level
    if thinking_level == ThinkingLevel.NA:
        _validate_anthropic_na(model=model)
        return
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        _validate_anthropic_adaptive(model=model)
        return
    if thinking_level == ThinkingLevel.BUDGET:
        _validate_anthropic_budget_for_provider(
            provider="anthropic", model=model, budget_tokens=reasoning.budget_tokens
        )
        return
    raise ValueError(
        f"Unsupported anthropic thinking level {thinking_level!r} for model={model!r}"
    )


def _validate_anthropic_na(*, model: str) -> None:
    if model in (
        set(ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED)
        | set(ANTHROPIC_BUDGET_THINKING_SUPPORTED)
    ):
        raise ValueError(
            f"thinking_level='na' is not supported for provider='anthropic' model={model!r}"
        )


def _validate_anthropic_adaptive(*, model: str) -> None:
    if model not in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        raise ValueError(
            f"anthropic adaptive thinking is not supported for model={model!r}"
        )


def validate_reasoning_for_kimi_code(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "kimi-code requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='budget', budget_tokens=...)"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise ValueError(
            f"kimi-code reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError("kimi-code does not support anthropic display controls")
    if reasoning.thinking_level not in {
        ThinkingLevel.NA,
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    }:
        raise ValueError(
            "kimi-code supports only anthropic thinking levels "
            "'na', 'off', 'adaptive', and 'budget'"
        )
    if reasoning.thinking_level == ThinkingLevel.BUDGET:
        _validate_anthropic_budget_for_provider(
            provider="kimi-code", model=model, budget_tokens=reasoning.budget_tokens
        )


def validate_reasoning_for_minimax(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        raise ValueError(
            f"reasoning is required for provider='minimax' model={model!r}"
        )
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "minimax requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='na')"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise ValueError(
            f"minimax reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError("minimax does not support anthropic display controls")
    if reasoning.thinking_level != ThinkingLevel.NA:
        raise ValueError(
            "minimax does not support explicit anthropic thinking; use thinking_level='na'"
        )


class AnthropicReasoningConfig(BaseProviderReasoningConfig):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> AnthropicReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(thinking={"type": "enabled", "budget_tokens": tokens})
            case AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.BUDGET:
                    tokens = require_budget_tokens(
                        budget_tokens, label="anthropic", min_value=1
                    )
                    thinking = {"type": "enabled", "budget_tokens": tokens}
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": "adaptive"}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message("anthropic", config)
                )


class KimiCodeReasoningConfig(BaseProviderReasoningConfig):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> KimiCodeReasoningConfig:
        if config is None:
            return cls()
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.OFF,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": "disabled"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": "adaptive"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=budget_tokens,
                display=None,
            ):
                tokens = require_budget_tokens(
                    budget_tokens, label="kimi-code", min_value=1
                )
                return cls(thinking={"type": "enabled", "budget_tokens": tokens})
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message("kimi-code", config)
                )


class MiniMaxReasoningConfig(BaseProviderReasoningConfig):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> MiniMaxReasoningConfig:
        if config is None:
            raise ProviderSemanticError(
                "minimax requires explicit AnthropicReasoning(thinking_level='na')"
            )
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message("minimax", config)
                )
