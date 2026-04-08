from __future__ import annotations

from dr_llm.llm.providers.reasoning_capability_types import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)

_GLM_THINKING_CAPS = ReasoningCapabilities(mode="glm")

GLM_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(model_prefix="glm-5", capabilities=_GLM_THINKING_CAPS),
    ReasoningCapabilityRule(model_prefix="glm-4.7", capabilities=_GLM_THINKING_CAPS),
    ReasoningCapabilityRule(model_prefix="glm-4.6", capabilities=_GLM_THINKING_CAPS),
    ReasoningCapabilityRule(model_prefix="glm-4.5", capabilities=_GLM_THINKING_CAPS),
)


def reasoning_capabilities_for_glm(model: str) -> ReasoningCapabilities | None:
    return resolve_capability_rules(GLM_CAPABILITY_RULES, model)
