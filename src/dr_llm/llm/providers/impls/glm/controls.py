from __future__ import annotations

from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    GlmReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
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


def validate_reasoning_for_glm(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    capabilities = reasoning_capabilities_for_glm(model)
    if reasoning is None:
        if not is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.GLM}' model={model!r}"
            )
        return
    if isinstance(reasoning, GlmReasoning):
        validate_allowed_thinking_levels(
            provider=ProviderName.GLM,
            model=model,
            thinking_level=reasoning.thinking_level,
            allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
            allow_na=False,
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.GLM}' model={model!r}; use GlmReasoning(thinking_level=...)"
        )
    raise ValueError(
        f"{ProviderName.GLM} reasoning is not supported for kind={reasoning.kind!r}"
    )


class GlmReasoningConfig(BaseProviderReasoningConfig):
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GlmReasoningConfig:
        if config is None:
            return cls()
        match config:
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(extra_body={"thinking": {"type": "disabled"}})
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(extra_body={"thinking": {"type": "enabled"}})
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.GLM, config)
        )
