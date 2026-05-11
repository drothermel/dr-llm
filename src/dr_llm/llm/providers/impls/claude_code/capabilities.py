from __future__ import annotations

from dr_llm.llm.names import EffortSpec, ReasoningMode
from dr_llm.llm.providers.impls.anthropic.effort import (
    supported_effort_levels_for_anthropic,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.impls.claude_code.families import (
    CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES,
)


def reasoning_capabilities_for_claude_code(
    model: str,
) -> ReasoningCapabilities | None:
    capability_rules = tuple(
        ReasoningCapabilityRule(
            family=family,
            capabilities=ReasoningCapabilities(
                mode=ReasoningMode.CLAUDE_CLI_EFFORT
            ),
        )
        for family in CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES
    )
    return resolve_capability_rules(capability_rules, model)


def supported_effort_levels_for_claude_code(
    model: str,
) -> tuple[EffortSpec, ...]:
    return supported_effort_levels_for_anthropic(model)
