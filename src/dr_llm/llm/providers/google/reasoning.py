from __future__ import annotations

from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.reasoning import (
    BaseProviderReasoningConfig,
    GoogleReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
    dispatch_reasoning_validation,
    google_literal_to_thinking_level,
    is_reasoning_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
    validate_budget_range,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    ReasoningCapabilities,
    reasoning_capabilities_for_model,
)

# Google Generative Language API `thinkingBudget` sentinel values.
_GOOGLE_THINKING_BUDGET_OFF = 0
_GOOGLE_THINKING_BUDGET_ADAPTIVE = -1


def validate_reasoning_for_google(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    capabilities = reasoning_capabilities_for_model(provider="google", model=model)

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        if capabilities is None:
            raise ValueError(
                f"Reasoning is not allowed for provider='google' model={model!r}: reasoning capabilities are unknown"
            )
        if capabilities.mode == "unsupported":
            raise ValueError(
                f"Reasoning is not supported for provider='google' model={model!r}"
            )
        if capabilities.mode == "google_level":
            raise ValueError(
                f"Top-level reasoning budget is not supported for provider='google' model={model!r} with capabilities.mode={capabilities.mode!r}"
            )
        validate_budget_range(
            provider="google",
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            capabilities=capabilities,
        )

    dispatch_reasoning_validation(
        provider="google",
        model=model,
        reasoning=reasoning,
        native_spec_type=GoogleReasoning,
        requires_reasoning=not is_reasoning_unsupported(capabilities),
        validate_native=lambda spec: _validate_google_reasoning_shape(
            model=model,
            thinking_level=spec.thinking_level,
            budget_tokens=spec.budget_tokens,
        ),
        validate_top_budget=_validate_top_budget,
    )


def _validate_google_reasoning_shape(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> None:
    capabilities = reasoning_capabilities_for_model(provider="google", model=model)
    if is_reasoning_unsupported(capabilities):
        if thinking_level == ThinkingLevel.NA:
            return
        raise ValueError(f"google thinking is not supported for model={model!r}")
    assert capabilities is not None
    if thinking_level == ThinkingLevel.NA:
        raise ValueError(
            f"thinking_level='na' is not supported for provider='google' model={model!r}"
        )
    if capabilities.mode == "google_budget":
        _validate_google_budget_mode(
            model=model,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            capabilities=capabilities,
        )
        return
    if capabilities.mode == "google_level":
        _validate_google_level_mode(
            model=model,
            thinking_level=thinking_level,
            capabilities=capabilities,
        )
        return
    raise ValueError(
        f"google reasoning is not supported for provider='google' model={model!r}"
    )


def _validate_google_budget_mode(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
    capabilities: ReasoningCapabilities,
) -> None:
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        if capabilities.supports_dynamic:
            return
        raise ValueError(
            f"google dynamic thinking is not supported for model={model!r}"
        )
    if thinking_level == ThinkingLevel.BUDGET:
        if budget_tokens is None:
            raise ValueError(
                "google budget thinking requires budget_tokens when "
                "thinking_level is 'budget'"
            )
        validate_budget_range(
            provider="google",
            model=model,
            label="google thinking_budget",
            tokens=budget_tokens,
            capabilities=capabilities,
        )
        return
    raise ValueError(
        f"google model {model!r} does not support thinking_level={thinking_level!r}; use off, adaptive, or budget"
    )


def _validate_google_level_mode(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    capabilities: ReasoningCapabilities,
) -> None:
    allowed_levels = {
        google_literal_to_thinking_level(level) for level in capabilities.google_levels
    }
    validate_allowed_thinking_levels(
        provider="google",
        model=model,
        thinking_level=thinking_level,
        allowed_levels=allowed_levels,
        allow_na=False,
    )


class GoogleReasoningConfig(BaseProviderReasoningConfig):
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GoogleReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(payload={"thinkingBudget": tokens})
            case GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                include_thoughts=include_thoughts,
            ):
                if thinking_level == ThinkingLevel.NA:
                    return cls()
                payload = _build_thinking_payload(
                    thinking_level=thinking_level,
                    budget_tokens=budget_tokens,
                )
                if include_thoughts is not None:
                    payload["includeThoughts"] = include_thoughts
                return cls(payload=payload)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message("google", config)
                )


_GOOGLE_LITERAL_LEVELS = {
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
}


def _build_thinking_payload(
    *,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> dict[str, Any]:
    if thinking_level == ThinkingLevel.OFF:
        return {"thinkingBudget": _GOOGLE_THINKING_BUDGET_OFF}
    if thinking_level == ThinkingLevel.ADAPTIVE:
        return {"thinkingBudget": _GOOGLE_THINKING_BUDGET_ADAPTIVE}
    if thinking_level == ThinkingLevel.BUDGET:
        return {
            "thinkingBudget": require_budget_tokens(
                budget_tokens, label="google", min_value=0
            )
        }
    if thinking_level in _GOOGLE_LITERAL_LEVELS:
        return {"thinkingLevel": str(thinking_level)}
    raise ProviderSemanticError(
        "google reasoning config did not contain a serializable setting"
    )
