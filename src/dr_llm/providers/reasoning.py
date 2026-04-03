from __future__ import annotations

from enum import StrEnum
from typing import Any, Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from dr_llm.providers.models import CallMode
from dr_llm.providers.reasoning_capabilities import (
    GoogleThinkingLevel,
    ReasoningCapabilities,
    reasoning_capabilities_for_model,
)


class ReasoningWarningCode(StrEnum):
    unsupported_for_provider = "unsupported_for_provider"
    mapped_with_heuristic = "mapped_with_heuristic"
    partially_supported = "partially_supported"


class ReasoningWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ReasoningWarningCode
    message: str
    provider: str | None = None
    mode: CallMode | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReasoningOff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["off"] = "off"

class ReasoningBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["budget"] = "budget"
    tokens: int

    @field_validator("tokens")
    @classmethod
    def _validate_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("reasoning budget tokens must be > 0")
        return value


class AnthropicReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["anthropic"] = "anthropic"
    budget_tokens: int | None = None
    thinking_mode: Literal["adaptive", "enabled"] | None = None
    display: Literal["summarized", "omitted"] | None = None

    @field_validator("budget_tokens")
    @classmethod
    def _validate_budget_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("anthropic reasoning budget_tokens must be > 0")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> AnthropicReasoning:
        if (
            self.budget_tokens is None
            and self.thinking_mode is None
            and self.display is None
        ):
            raise ValueError(
                "anthropic reasoning requires at least one configured field"
            )
        if self.thinking_mode == "adaptive" and self.budget_tokens is not None:
            raise ValueError(
                "anthropic adaptive thinking does not accept budget_tokens"
            )
        if self.thinking_mode == "enabled" and self.budget_tokens is None:
            raise ValueError("anthropic manual thinking requires budget_tokens")
        if (
            self.display is not None
            and self.budget_tokens is None
            and self.thinking_mode is None
        ):
            raise ValueError(
                "anthropic display requires thinking configuration, not effort-only output control"
            )
        return self


class GoogleReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["google"] = "google"
    thinking_level: GoogleThinkingLevel | None = None
    thinking_budget: int | None = None
    dynamic: bool | None = None

    @field_validator("thinking_budget")
    @classmethod
    def _validate_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("google thinking_budget must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> GoogleReasoning:
        configured = [
            self.thinking_level is not None,
            self.thinking_budget is not None,
            self.dynamic is True,
        ]
        if sum(configured) != 1:
            raise ValueError(
                "google reasoning requires exactly one of thinking_level, thinking_budget, or dynamic=true"
            )
        return self


ReasoningSpec = Annotated[
    ReasoningOff
    | ReasoningBudget
    | AnthropicReasoning
    | GoogleReasoning,
    Field(discriminator="kind"),
]
REASONING_SPEC_ADAPTER = TypeAdapter(ReasoningSpec)


def parse_reasoning_spec(payload: object) -> ReasoningSpec:
    return REASONING_SPEC_ADAPTER.validate_python(payload)


def validate_reasoning(
    *,
    provider: str,
    model: str,
    reasoning: ReasoningSpec | None,
) -> None:
    if reasoning is None:
        return
    capabilities = reasoning_capabilities_for_model(provider=provider, model=model)
    if capabilities is None:
        raise ValueError(
            f"Reasoning is not allowed for provider={provider!r} model={model!r}: reasoning capabilities are unknown"
        )
    _validate_against_capabilities(
        provider=provider,
        model=model,
        reasoning=reasoning,
        capabilities=capabilities,
    )


def _validate_against_capabilities(
    *,
    provider: str,
    model: str,
    reasoning: ReasoningSpec,
    capabilities: ReasoningCapabilities,
) -> None:
    if capabilities.mode in {"unsupported", "codex_headless"}:
        raise ValueError(
            f"Reasoning is not supported for provider={provider!r} model={model!r}"
        )

    match reasoning:
        case ReasoningOff():
            if not capabilities.supports_off:
                raise ValueError(
                    f"reasoning kind='off' is not supported for provider={provider!r} model={model!r}"
                )
        case ReasoningBudget(tokens=tokens):
            _validate_budget_range(
                provider=provider,
                model=model,
                label="reasoning budget",
                tokens=tokens,
                capabilities=capabilities,
            )
        case GoogleReasoning(
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            dynamic=dynamic,
        ):
            if capabilities.mode == "google_level":
                if thinking_level is None:
                    raise ValueError(
                        f"google model {model!r} requires thinking_level-based reasoning"
                    )
                if thinking_level not in capabilities.google_levels:
                    raise ValueError(
                        f"google thinking level {thinking_level!r} is not supported for model={model!r}"
                    )
                return
            if capabilities.mode != "google_budget":
                raise ValueError(
                    f"google reasoning is not supported for provider={provider!r} model={model!r}"
                )
            if thinking_level is not None:
                raise ValueError(
                    f"google model {model!r} does not support thinking_level; use thinking_budget or dynamic"
                )
            if dynamic:
                if not capabilities.supports_dynamic:
                    raise ValueError(
                        f"google dynamic reasoning is not supported for model={model!r}"
                    )
                return
            assert thinking_budget is not None
            _validate_budget_range(
                provider=provider,
                model=model,
                label="google thinking_budget",
                tokens=thinking_budget,
                capabilities=capabilities,
            )
        case AnthropicReasoning(
            budget_tokens=budget_tokens,
            thinking_mode=thinking_mode,
            display=display,
        ):
            if capabilities.mode not in {"anthropic_budget", "anthropic_effort_and_budget"}:
                raise ValueError(
                    f"anthropic reasoning is not supported for provider={provider!r} model={model!r}"
                )
            if budget_tokens is not None:
                _validate_budget_range(
                    provider=provider,
                    model=model,
                    label="anthropic budget_tokens",
                    tokens=budget_tokens,
                    capabilities=capabilities,
                )
            if display is not None and not capabilities.supports_display:
                raise ValueError(
                    f"anthropic display controls are not supported for model={model!r}"
                )
            if thinking_mode == "adaptive" and not capabilities.supports_adaptive:
                raise ValueError(
                    f"anthropic adaptive thinking is not supported for model={model!r}"
                )
            if thinking_mode == "enabled" and capabilities.min_budget_tokens is None:
                raise ValueError(
                    f"anthropic manual thinking is not supported for model={model!r}"
                )
        case _:
            raise ValueError(
                f"Unsupported reasoning configuration for provider={provider!r} model={model!r}"
            )


def _validate_budget_range(
    *,
    provider: str,
    model: str,
    label: str,
    tokens: int,
    capabilities: ReasoningCapabilities,
) -> None:
    if capabilities.min_budget_tokens is None or capabilities.max_budget_tokens is None:
        raise ValueError(
            f"{label} is not supported for provider={provider!r} model={model!r}"
        )
    if (
        tokens < capabilities.min_budget_tokens
        or tokens > capabilities.max_budget_tokens
    ):
        raise ValueError(
            f"{label} must be between {capabilities.min_budget_tokens} and {capabilities.max_budget_tokens} for provider={provider!r} model={model!r}"
        )
