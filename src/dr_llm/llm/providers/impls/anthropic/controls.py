from __future__ import annotations

from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    is_reasoning_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_budget_range,
)
from dr_llm.llm.providers.impls.anthropic.families import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
    ANTHROPIC_BUDGET_CAPABILITY_FAMILIES,
    ANTHROPIC_BUDGET_THINKING_SUPPORTED,
    AnthropicModelFamily,
)

ANTHROPIC_BUDGET_MIN_TOKENS = 1024
ANTHROPIC_BUDGET_MAX_TOKENS = 128000

_ANTHROPIC_OPUS_46_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        family=AnthropicModelFamily.CLAUDE_OPUS_46,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.ANTHROPIC_EFFORT
        ),
    ),
)
_ANTHROPIC_SONNET_46_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        family=AnthropicModelFamily.CLAUDE_SONNET_46,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.ANTHROPIC_EFFORT
        ),
    ),
)
_ANTHROPIC_OPUS_45_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        family=AnthropicModelFamily.CLAUDE_OPUS_45,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET,
            min_budget_tokens=ANTHROPIC_BUDGET_MIN_TOKENS,
            max_budget_tokens=ANTHROPIC_BUDGET_MAX_TOKENS,
        ),
    ),
)
_ANTHROPIC_BUDGET_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.ANTHROPIC_BUDGET,
            min_budget_tokens=ANTHROPIC_BUDGET_MIN_TOKENS,
            max_budget_tokens=ANTHROPIC_BUDGET_MAX_TOKENS,
        ),
    )
    for family in ANTHROPIC_BUDGET_CAPABILITY_FAMILIES
)


def reasoning_capabilities_for_anthropic(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(
        (
            *_ANTHROPIC_OPUS_46_RULES,
            *_ANTHROPIC_SONNET_46_RULES,
            *_ANTHROPIC_OPUS_45_RULES,
            *_ANTHROPIC_BUDGET_RULES,
        ),
        model,
    )


def anthropic_reasoning_mode(model: str) -> ReasoningMode:
    capabilities = reasoning_capabilities_for_anthropic(model)
    if capabilities is None:
        return ReasoningMode.UNSUPPORTED
    return capabilities.mode


def anthropic_supports_adaptive_thinking(model: str) -> bool:
    return anthropic_reasoning_mode(model) == ReasoningMode.ANTHROPIC_EFFORT


def anthropic_supports_budget_thinking(model: str) -> bool:
    return anthropic_reasoning_mode(model) in {
        ReasoningMode.ANTHROPIC_BUDGET,
        ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET,
    }


def anthropic_supports_effort(model: str) -> bool:
    return anthropic_reasoning_mode(model) in {
        ReasoningMode.ANTHROPIC_EFFORT,
        ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET,
    }


def supported_effort_levels_for_anthropic(
    model: str,
) -> tuple[EffortSpec, ...]:
    if not anthropic_supports_effort(model):
        return ()
    levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    if AnthropicModelFamily.CLAUDE_OPUS_46.in_family(model):
        levels.append(EffortSpec.MAX)
    return tuple(levels)


def validate_anthropic_budget_for_provider(
    *,
    provider: str,
    model: str,
    budget_tokens: int | None,
    capabilities: ReasoningCapabilities | None,
) -> None:
    if budget_tokens is None:
        raise ValueError(
            f"{provider} budget thinking requires budget_tokens when "
            "thinking_level is 'budget'"
        )
    unsupported_anthropic_model = (
        provider == ProviderName.ANTHROPIC
        and not anthropic_supports_budget_thinking(model)
    )
    if (
        unsupported_anthropic_model
        or capabilities is None
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
    capabilities = reasoning_capabilities_for_anthropic(model)
    dispatch_reasoning_validation(
        provider=ProviderName.ANTHROPIC,
        model=model,
        reasoning=reasoning,
        native_spec_type=AnthropicReasoning,
        requires_reasoning=not is_reasoning_unsupported(capabilities),
        validate_native=lambda spec: _validate_anthropic_reasoning_shape(
            model=model, reasoning=spec, capabilities=capabilities
        ),
        validate_top_budget=lambda budget: (
            validate_anthropic_budget_for_provider(
                provider=ProviderName.ANTHROPIC,
                model=model,
                budget_tokens=budget.tokens,
                capabilities=capabilities,
            )
        ),
    )


def _validate_anthropic_reasoning_shape(
    *,
    model: str,
    reasoning: AnthropicReasoning,
    capabilities: ReasoningCapabilities | None,
) -> None:
    thinking_level = reasoning.thinking_level
    if (
        thinking_level != ThinkingLevel.BUDGET
        and reasoning.budget_tokens is not None
    ):
        raise ValueError(
            "anthropic budget_tokens are only allowed with thinking_level='budget'"
        )
    if thinking_level == ThinkingLevel.NA:
        _validate_anthropic_na(model=model)
        return
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        _validate_anthropic_adaptive(model=model)
        return
    if thinking_level == ThinkingLevel.BUDGET:
        validate_anthropic_budget_for_provider(
            provider=ProviderName.ANTHROPIC,
            model=model,
            budget_tokens=reasoning.budget_tokens,
            capabilities=capabilities,
        )
        return
    raise ValueError(
        f"Unsupported anthropic thinking level {thinking_level!r} for model={model!r}"
    )


def _validate_anthropic_na(*, model: str) -> None:
    if not is_reasoning_unsupported(
        reasoning_capabilities_for_anthropic(model)
    ):
        raise ValueError(
            f"thinking_level='na' is not supported for provider='{ProviderName.ANTHROPIC}' model={model!r}"
        )


def _validate_anthropic_adaptive(*, model: str) -> None:
    if not anthropic_supports_adaptive_thinking(model):
        raise ValueError(
            f"anthropic adaptive thinking is not supported for model={model!r}"
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
                return cls(
                    thinking={"type": "enabled", "budget_tokens": tokens}
                )
            case AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.BUDGET:
                    tokens = require_budget_tokens(
                        budget_tokens,
                        label=ProviderName.ANTHROPIC,
                        min_value=1,
                    )
                    thinking = {"type": "enabled", "budget_tokens": tokens}
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": "adaptive"}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.ANTHROPIC, config
                    )
                )


ANTHROPIC_THINKING_MIN_BUDGET_TOKENS = ANTHROPIC_BUDGET_MIN_TOKENS
ANTHROPIC_THINKING_MAX_BUDGET_TOKENS = ANTHROPIC_BUDGET_MAX_TOKENS

__all__ = [
    "ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED",
    "ANTHROPIC_BUDGET_THINKING_SUPPORTED",
    "ANTHROPIC_THINKING_MAX_BUDGET_TOKENS",
    "ANTHROPIC_THINKING_MIN_BUDGET_TOKENS",
]
