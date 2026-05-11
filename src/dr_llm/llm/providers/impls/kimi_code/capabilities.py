from __future__ import annotations

from dr_llm.llm.names import EffortSpec, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT
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
