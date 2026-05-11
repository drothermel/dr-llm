from __future__ import annotations

from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    GlmReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    validate_allowed_thinking_levels,
)
from dr_llm.llm.providers.glm.capabilities import (
    reasoning_capabilities_for_glm,
)


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
