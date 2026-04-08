from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.llm.providers.reasoning import (
    BaseProviderReasoningConfig,
    GlmReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
    dispatch_reasoning_validation,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    reasoning_capabilities_for_model,
)


def validate_reasoning_for_openai(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: OpenAIReasoning) -> None:
        if not openai_supports_configurable_thinking(model):
            raise ValueError(f"openai thinking is not supported for model={model!r}")
        validate_discrete_thinking_level(
            provider="openai",
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        del budget
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='openai' model={model!r}; use OpenAIReasoning(thinking_level=...)"
        )

    dispatch_reasoning_validation(
        provider="openai",
        model=model,
        reasoning=reasoning,
        native_spec_type=OpenAIReasoning,
        requires_reasoning=openai_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


def validate_reasoning_for_openrouter(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if openrouter_model_policy(model) is None:
        raise ValueError(f"openrouter model={model!r} is not in the curated allowlist")
    capabilities = reasoning_capabilities_for_model(provider="openrouter", model=model)
    if reasoning is None:
        if not is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"reasoning is required for provider='openrouter' model={model!r}"
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
            raise ValueError(f"openai thinking is not supported for model={model!r}")
        validate_discrete_thinking_level(
            provider="openrouter",
            model=model,
            thinking_level=reasoning.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='openrouter' model={model!r}; use OpenRouterReasoning or OpenAIReasoning"
        )
    raise ValueError(
        f"openrouter reasoning is not supported for kind={reasoning.kind!r}"
    )


def _validate_openrouter_shape(
    *,
    model: str,
    enabled: bool | None,
    effort: str | None,
) -> None:
    policy = openrouter_model_policy(model)
    if policy is None:
        raise ValueError(f"openrouter reasoning is not supported for model={model!r}")
    if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
        raise ValueError(f"openrouter reasoning is not supported for model={model!r}")
    if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        if effort is not None:
            raise ValueError(
                f"openrouter effort controls are not supported for model={model!r}"
            )
        if enabled is None:
            raise ValueError(
                f"openrouter reasoning requires the enabled flag for model={model!r}"
            )
        if not enabled and not policy.supports_disable:
            raise ValueError(
                f"openrouter reasoning cannot be disabled for model={model!r}"
            )
        return
    if enabled is not None:
        raise ValueError(
            f"openrouter enabled controls are not supported for model={model!r}"
        )
    if effort is None:
        raise ValueError(
            f"openrouter reasoning requires an effort level for model={model!r}"
        )
    if effort not in policy.allowed_efforts:
        allowed = ", ".join(policy.allowed_efforts)
        raise ValueError(
            f"openrouter effort={effort!r} is not supported for model={model!r}; allowed levels: {allowed}"
        )


def validate_reasoning_for_glm(*, model: str, reasoning: ReasoningSpec | None) -> None:
    capabilities = reasoning_capabilities_for_model(provider="glm", model=model)
    if reasoning is None:
        if not is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"reasoning is required for provider='glm' model={model!r}"
            )
        return
    if isinstance(reasoning, GlmReasoning):
        validate_allowed_thinking_levels(
            provider="glm",
            model=model,
            thinking_level=reasoning.thinking_level,
            allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
            allow_na=False,
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='glm' model={model!r}; use GlmReasoning(thinking_level=...)"
        )
    raise ValueError(f"glm reasoning is not supported for kind={reasoning.kind!r}")


class OpenAICompatReasoningConfig(BaseProviderReasoningConfig):
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> OpenAICompatReasoningConfig:
        if config is None:
            return cls()
        match config:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort="none")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort="minimal")
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort="low")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort="medium")
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort="high")
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(extra_body={"thinking": {"type": "disabled"}})
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(extra_body={"thinking": {"type": "enabled"}})
            case OpenRouterReasoning(enabled=enabled, effort=effort):
                if provider != "openrouter":
                    raise ProviderSemanticError(
                        "OpenRouter reasoning serializer requires provider='openrouter'"
                    )
                reasoning_payload: dict[str, Any]
                if enabled is not None:
                    reasoning_payload = {"enabled": enabled}
                elif effort is not None:
                    reasoning_payload = {"effort": effort}
                else:
                    raise ProviderSemanticError(
                        f"OpenRouter reasoning serializer received invalid config for model={model!r}"
                    )
                return cls(extra_body={"reasoning": reasoning_payload})
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message("OpenAI-compatible", config)
        )
