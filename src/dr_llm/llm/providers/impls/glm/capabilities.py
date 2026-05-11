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

GLM_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    *(
        ReasoningCapabilityRule(
            family=family,
            capabilities=ReasoningCapabilities(mode=ReasoningMode.GLM),
        )
        for family in GLM_THINKING_SUPPORTED_FAMILIES
    ),
)


def reasoning_capabilities_for_glm(model: str) -> ReasoningCapabilities | None:
    return resolve_capability_rules(GLM_CAPABILITY_RULES, model)
