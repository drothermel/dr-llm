from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.impls.anthropic.families import (
    ANTHROPIC_BUDGET_CAPABILITY_FAMILIES,
    AnthropicModelFamily,
)

ANTHROPIC_BUDGET_MIN_TOKENS = 1024
ANTHROPIC_BUDGET_MAX_TOKENS = 128000

_ANTHROPIC_BUDGET_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_BUDGET,
    min_budget_tokens=ANTHROPIC_BUDGET_MIN_TOKENS,
    max_budget_tokens=ANTHROPIC_BUDGET_MAX_TOKENS,
)
_ANTHROPIC_SONNET_46_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT
)
_ANTHROPIC_OPUS_45_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET,
    min_budget_tokens=ANTHROPIC_BUDGET_MIN_TOKENS,
    max_budget_tokens=ANTHROPIC_BUDGET_MAX_TOKENS,
)
_ANTHROPIC_OPUS_46_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT
)

ANTHROPIC_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        model_prefix=AnthropicModelFamily.CLAUDE_OPUS_46,
        capabilities=_ANTHROPIC_OPUS_46_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix=AnthropicModelFamily.CLAUDE_SONNET_46,
        capabilities=_ANTHROPIC_SONNET_46_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix=AnthropicModelFamily.CLAUDE_OPUS_45,
        capabilities=_ANTHROPIC_OPUS_45_CAPS,
    ),
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_ANTHROPIC_BUDGET_CAPS
        )
        for family in ANTHROPIC_BUDGET_CAPABILITY_FAMILIES
    ),
)


def reasoning_capabilities_for_anthropic(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(ANTHROPIC_CAPABILITY_RULES, model)


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
