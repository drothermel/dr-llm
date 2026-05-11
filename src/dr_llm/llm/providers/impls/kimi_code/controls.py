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
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
)
from dr_llm.llm.providers.impls.anthropic.controls import (
    validate_anthropic_budget_for_provider,
)
from dr_llm.llm.providers.impls.kimi_code.families import (
    KimiCodeModelFamily,
)


def reasoning_capabilities_for_kimi_code(
    model: str,
) -> ReasoningCapabilities | None:
    capability_rules = (
        ReasoningCapabilityRule(
            family=KimiCodeModelFamily.KIMI_FOR_CODING,
            capabilities=ReasoningCapabilities(
                mode=ReasoningMode.KIMI_CODE_EFFORT_AND_BUDGET,
                min_budget_tokens=1024,
                max_budget_tokens=128000,
            ),
        ),
    )
    return resolve_capability_rules(capability_rules, model)


def supported_effort_levels_for_kimi_code(
    model: str,
) -> tuple[EffortSpec, ...]:
    if reasoning_capabilities_for_kimi_code(model) is None:
        return ()
    return FULL_EFFORT


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
        raise ValueError(
            "kimi-code does not support anthropic display controls"
        )
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
        validate_anthropic_budget_for_provider(
            provider=ProviderName.KIMI_CODE,
            model=model,
            budget_tokens=reasoning.budget_tokens,
            capabilities=reasoning_capabilities_for_kimi_code(model),
        )
        return
    if reasoning.budget_tokens is not None:
        raise ValueError(
            "kimi-code budget_tokens are only valid when "
            "thinking_level is 'budget'"
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
                    budget_tokens, label=ProviderName.KIMI_CODE, min_value=1
                )
                return cls(
                    thinking={"type": "enabled", "budget_tokens": tokens}
                )
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.KIMI_CODE, config
                    )
                )
