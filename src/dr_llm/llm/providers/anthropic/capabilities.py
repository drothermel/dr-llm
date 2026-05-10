from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)

_ANTHROPIC_BUDGET_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_BUDGET,
    min_budget_tokens=1024,
    max_budget_tokens=128000,
)
_ANTHROPIC_SONNET_46_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT
)
_ANTHROPIC_OPUS_45_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET,
    min_budget_tokens=1024,
    max_budget_tokens=128000,
)
_ANTHROPIC_OPUS_46_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.ANTHROPIC_EFFORT
)

ANTHROPIC_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        model_prefix="claude-opus-4-6", capabilities=_ANTHROPIC_OPUS_46_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-sonnet-4-6",
        capabilities=_ANTHROPIC_SONNET_46_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-opus-4-5", capabilities=_ANTHROPIC_OPUS_45_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-opus-4-1", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-opus-4-", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-sonnet-4-5", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-sonnet-4-", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-3-7-sonnet", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
    ReasoningCapabilityRule(
        model_prefix="claude-haiku-4-5", capabilities=_ANTHROPIC_BUDGET_CAPS
    ),
)


def reasoning_capabilities_for_anthropic(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(ANTHROPIC_CAPABILITY_RULES, model)
