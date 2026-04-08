from __future__ import annotations

from dr_llm.llm.providers.effort_types import FULL_EFFORT, EffortSpec
from dr_llm.llm.providers.reasoning_capability_types import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)

_MINIMAX_CAPS = ReasoningCapabilities(mode="minimax_effort")

MINIMAX_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(model_prefix="MiniMax-", capabilities=_MINIMAX_CAPS),
)


def reasoning_capabilities_for_minimax(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(MINIMAX_CAPABILITY_RULES, model)


def supported_effort_levels_for_minimax(model: str) -> tuple[EffortSpec, ...]:
    if reasoning_capabilities_for_minimax(model) is None:
        return ()
    return FULL_EFFORT
