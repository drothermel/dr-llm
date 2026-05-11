from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.impls.glm.families import (
    GLM_THINKING_SUPPORTED_FAMILIES,
)


def reasoning_capabilities_for_glm(model: str) -> ReasoningCapabilities | None:
    capability_rules = tuple(
        ReasoningCapabilityRule(
            family=family,
            capabilities=ReasoningCapabilities(mode=ReasoningMode.GLM),
        )
        for family in GLM_THINKING_SUPPORTED_FAMILIES
    )
    return resolve_capability_rules(capability_rules, model)
