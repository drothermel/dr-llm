from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    GoogleThinkingLevel,
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    GoogleReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    google_literal_to_thinking_level,
    is_reasoning_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
    validate_budget_range,
)
from dr_llm.llm.providers.impls.google.families import (
    GEMMA_4_FAMILIES,
    GOOGLE_25_FLASH_FAMILIES,
    GOOGLE_25_FLASH_LITE_FAMILIES,
    GOOGLE_25_PRO_FAMILIES,
    GOOGLE_3_FAMILIES,
)


class GoogleMinBudget(IntEnum):
    GEMINI_25_FLASH = 1
    GEMINI_25_FLASH_LITE = 512
    GEMINI_25_PRO = 128


class GoogleMaxBudget(IntEnum):
    GEMINI_25_FLASH = 24576
    GEMINI_25_FLASH_LITE = 24576
    GEMINI_25_PRO = 32768


_GOOGLE_25_FLASH_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_FLASH_FAMILIES
)
_GOOGLE_25_FLASH_LITE_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH_LITE,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH_LITE,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_FLASH_LITE_FAMILIES
)
_GOOGLE_25_PRO_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_PRO,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_PRO,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_PRO_FAMILIES
)
_GOOGLE_3_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_LEVEL,
            google_thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.LOW,
                GoogleThinkingLevel.MEDIUM,
                GoogleThinkingLevel.HIGH,
            ),
        ),
    )
    for family in GOOGLE_3_FAMILIES
)
_GEMMA_4_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_LEVEL,
            google_thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.HIGH,
            ),
        ),
    )
    for family in GEMMA_4_FAMILIES
)


def reasoning_capabilities_for_google(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(
        (
            *_GOOGLE_25_FLASH_LITE_RULES,
            *_GOOGLE_25_FLASH_RULES,
            *_GOOGLE_25_PRO_RULES,
            *_GOOGLE_3_RULES,
            *_GEMMA_4_RULES,
        ),
        model,
    )


# Google Generative Language API `thinkingBudget` sentinel values.
_GOOGLE_THINKING_BUDGET_OFF = 0
_GOOGLE_THINKING_BUDGET_ADAPTIVE = -1


def validate_reasoning_for_google(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    capabilities = reasoning_capabilities_for_google(model)

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        if capabilities is None:
            raise ValueError(
                f"Reasoning is not allowed for provider='{ProviderName.GOOGLE}' model={model!r}: reasoning capabilities are unknown"
            )
        if capabilities.mode == ReasoningMode.UNSUPPORTED:
            raise ValueError(
                f"Reasoning is not supported for provider='{ProviderName.GOOGLE}' model={model!r}"
            )
        if capabilities.mode == ReasoningMode.GOOGLE_LEVEL:
            raise ValueError(
                f"Top-level reasoning budget is not supported for provider='{ProviderName.GOOGLE}' model={model!r} with capabilities.mode={capabilities.mode!r}"
            )
        validate_budget_range(
            provider=ProviderName.GOOGLE,
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            capabilities=capabilities,
        )

    dispatch_reasoning_validation(
        provider=ProviderName.GOOGLE,
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
    capabilities = reasoning_capabilities_for_google(model)
    if is_reasoning_unsupported(capabilities):
        if thinking_level == ThinkingLevel.NA:
            return
        raise ValueError(
            f"{ProviderName.GOOGLE} thinking is not supported for model={model!r}"
        )
    assert capabilities is not None
    if thinking_level == ThinkingLevel.NA:
        raise ValueError(
            f"thinking_level='na' is not supported for provider='{ProviderName.GOOGLE}' model={model!r}"
        )
    if capabilities.mode == ReasoningMode.GOOGLE_BUDGET:
        _validate_google_budget_mode(
            model=model,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            capabilities=capabilities,
        )
        return
    if capabilities.mode == ReasoningMode.GOOGLE_LEVEL:
        _validate_google_level_mode(
            model=model,
            thinking_level=thinking_level,
            capabilities=capabilities,
        )
        return
    raise ValueError(
        f"Reasoning is not supported for provider='{ProviderName.GOOGLE}' model={model!r}"
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
            f"{ProviderName.GOOGLE} dynamic thinking is not supported for model={model!r}"
        )
    if thinking_level == ThinkingLevel.BUDGET:
        if budget_tokens is None:
            raise ValueError(
                "google budget thinking requires budget_tokens when "
                "thinking_level is 'budget'"
            )
        validate_budget_range(
            provider=ProviderName.GOOGLE,
            model=model,
            label=f"{ProviderName.GOOGLE} thinking_budget",
            tokens=budget_tokens,
            capabilities=capabilities,
        )
        return
    raise ValueError(
        f"{ProviderName.GOOGLE} model {model!r} does not support thinking_level={thinking_level!r}; use off, adaptive, or budget"
    )


def _validate_google_level_mode(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    capabilities: ReasoningCapabilities,
) -> None:
    allowed_levels = {
        google_literal_to_thinking_level(level)
        for level in capabilities.google_thinking_levels
    }
    validate_allowed_thinking_levels(
        provider=ProviderName.GOOGLE,
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
                    unsupported_reasoning_kind_message(
                        ProviderName.GOOGLE, config
                    )
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
                budget_tokens, label=ProviderName.GOOGLE, min_value=0
            )
        }
    if thinking_level in _GOOGLE_LITERAL_LEVELS:
        return {"thinkingLevel": str(thinking_level)}
    raise ProviderSemanticError(
        f"{ProviderName.GOOGLE} reasoning config did not contain a serializable setting"
    )
