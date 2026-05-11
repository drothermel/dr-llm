from __future__ import annotations

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.openai.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)


def validate_reasoning_for_openai(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: OpenAIReasoning) -> None:
        if not openai_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENAI,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        del budget
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.OPENAI}' model={model!r}; use OpenAIReasoning(thinking_level=...)"
        )

    dispatch_reasoning_validation(
        provider=ProviderName.OPENAI,
        model=model,
        reasoning=reasoning,
        native_spec_type=OpenAIReasoning,
        requires_reasoning=openai_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )
