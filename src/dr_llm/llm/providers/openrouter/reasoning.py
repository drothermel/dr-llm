from __future__ import annotations

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.openai.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
    reasoning_capabilities_for_openrouter,
)


def validate_reasoning_for_openrouter(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if openrouter_model_policy(model) is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} model={model!r} is not in the curated allowlist"
        )
    capabilities = reasoning_capabilities_for_openrouter(model)
    if reasoning is None:
        if not is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.OPENROUTER}' model={model!r}"
            )
        return
    if isinstance(reasoning, OpenRouterReasoning):
        _validate_openrouter_shape(
            model=model,
            enabled=reasoning.enabled,
            effort=reasoning.effort,
        )
        return
    if isinstance(reasoning, OpenAIReasoning):
        if not openai_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENROUTER,
            model=model,
            thinking_level=reasoning.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.OPENROUTER}' model={model!r}; use OpenRouterReasoning or OpenAIReasoning"
        )
    raise ValueError(
        f"{ProviderName.OPENROUTER} reasoning is not supported for kind={reasoning.kind!r}"
    )


def _validate_openrouter_shape(
    *,
    model: str,
    enabled: bool | None,
    effort: str | None,
) -> None:
    policy = openrouter_model_policy(model)
    if policy is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        if effort is not None:
            raise ValueError(
                f"{ProviderName.OPENROUTER} effort controls are not supported for model={model!r}"
            )
        if enabled is None:
            raise ValueError(
                f"{ProviderName.OPENROUTER} reasoning requires the enabled flag for model={model!r}"
            )
        if not enabled and not policy.supports_disable:
            raise ValueError(
                f"{ProviderName.OPENROUTER} reasoning cannot be disabled for model={model!r}"
            )
        return
    if enabled is not None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} enabled controls are not supported for model={model!r}"
        )
    if effort is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning requires an effort level for model={model!r}"
        )
    if effort not in policy.allowed_efforts:
        allowed = ", ".join(policy.allowed_efforts)
        raise ValueError(
            f"{ProviderName.OPENROUTER} effort={effort!r} is not supported for model={model!r}; allowed levels: {allowed}"
        )
