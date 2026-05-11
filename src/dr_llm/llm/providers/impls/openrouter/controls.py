from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import (
    OpenRouterEffortLevel,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.impls.openai.controls import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)

if TYPE_CHECKING:
    from dr_llm.llm.catalog.models import ModelCatalogEntry


class OpenRouterReasoningRequestStyle(StrEnum):
    NONE = "none"
    ENABLED_FLAG = "enabled_flag"
    EFFORT = "effort"


class OpenRouterModelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    request_style: OpenRouterReasoningRequestStyle
    supports_disable: bool
    allowed_efforts: tuple[OpenRouterEffortLevel, ...] = ()
    default_enabled: bool | None = None
    verified: bool = False
    notes: str | None = None


@cache
def _policies() -> dict[str, OpenRouterModelPolicy]:
    raw = yaml.safe_load(
        files("dr_llm.llm.providers.impls.openrouter.data")
        .joinpath("model_policies.yml")
        .read_text(encoding="utf-8")
    )
    return {
        model: OpenRouterModelPolicy(model=model, **fields)
        for model, fields in raw.items()
    }


def openrouter_model_policy(model: str) -> OpenRouterModelPolicy | None:
    return _policies().get(model)


def reasoning_capabilities_for_openrouter(
    model: str,
) -> ReasoningCapabilities | None:
    policy = openrouter_model_policy(model)
    if policy is None:
        return None
    return _capabilities_for_policy(policy.request_style)


def openrouter_allowed_models() -> tuple[str, ...]:
    return tuple(_policies())


def apply_openrouter_model_policies(
    entries: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    filtered: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.provider != ProviderName.OPENROUTER:
            filtered.append(entry)
            continue
        policy = openrouter_model_policy(entry.model)
        if policy is None:
            continue
        capabilities = _capabilities_for_policy(policy.request_style)
        filtered.append(
            entry.model_copy(
                update={
                    "supports_reasoning": capabilities.supports_reasoning,
                    "reasoning_capabilities": capabilities,
                }
            )
        )
    return filtered


def _capabilities_for_policy(
    request_style: OpenRouterReasoningRequestStyle,
) -> ReasoningCapabilities:
    if request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        return ReasoningCapabilities(mode=ReasoningMode.OPENROUTER_TOGGLE)
    if request_style == OpenRouterReasoningRequestStyle.EFFORT:
        return ReasoningCapabilities(mode=ReasoningMode.OPENROUTER_EFFORT)
    return ReasoningCapabilities(mode=ReasoningMode.UNSUPPORTED)


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


class OpenRouterReasoningConfig(BaseProviderReasoningConfig):
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high"] | None
    ) = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> OpenRouterReasoningConfig:
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
            case OpenRouterReasoning(enabled=enabled, effort=effort):
                reasoning_payload: dict[str, Any]
                if enabled is not None:
                    reasoning_payload = {"enabled": enabled}
                elif effort is not None:
                    reasoning_payload = {"effort": effort}
                else:
                    raise ProviderSemanticError(
                        "OpenRouter reasoning serializer received invalid config"
                    )
                return cls(extra_body={"reasoning": reasoning_payload})
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.OPENROUTER, config)
        )
