from __future__ import annotations

from dr_llm.llm.names import EffortSpec, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT
from dr_llm.llm.providers.impls.minimax.families import (
    MINIMAX_SUPPORTED_MODEL_FAMILIES,
)

MINIMAX_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    *(
        ReasoningCapabilityRule(
            model_prefix=family,
            capabilities=ReasoningCapabilities(
                mode=ReasoningMode.MINIMAX_EFFORT
            ),
        )
        for family in MINIMAX_SUPPORTED_MODEL_FAMILIES
    ),
)


def reasoning_capabilities_for_minimax(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(MINIMAX_CAPABILITY_RULES, model)


def supported_effort_levels_for_minimax(model: str) -> tuple[EffortSpec, ...]:
    if reasoning_capabilities_for_minimax(model) is None:
        return ()
    return FULL_EFFORT
