from __future__ import annotations

from collections.abc import Callable
from typing import Any, Annotated, Literal, Never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from dr_llm.llm.names import (
    OpenRouterEffortLevel,
    ProviderName,
    ControlMode,
    ReasoningWarningCode,
    ThinkingLevel,
)


class ReasoningWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ReasoningWarningCode
    message: str
    provider: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


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

    kind: Literal[ProviderName.ANTHROPIC] = ProviderName.ANTHROPIC
    thinking_level: ThinkingLevel = ThinkingLevel.NA
    budget_tokens: int | None = None
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
            self.thinking_level == ThinkingLevel.BUDGET
            and self.budget_tokens is None
        ):
            raise ValueError(
                "anthropic budget thinking requires budget_tokens"
            )
        if (
            self.thinking_level != ThinkingLevel.BUDGET
            and self.budget_tokens is not None
        ):
            raise ValueError(
                "anthropic budget_tokens are only allowed with thinking_level='budget'"
            )
        if self.display is not None and self.thinking_level not in {
            ThinkingLevel.BUDGET,
            ThinkingLevel.ADAPTIVE,
        }:
            raise ValueError(
                "anthropic display requires thinking_level='budget' or thinking_level='adaptive'"
            )
        return self


class OpenAIReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderName.OPENAI] = ProviderName.OPENAI
    thinking_level: ThinkingLevel = ThinkingLevel.NA


class CodexReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderName.CODEX] = ProviderName.CODEX
    thinking_level: ThinkingLevel = ThinkingLevel.NA


class GlmReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderName.GLM] = ProviderName.GLM
    thinking_level: ThinkingLevel = ThinkingLevel.NA

    @model_validator(mode="after")
    def _validate_shape(self) -> GlmReasoning:
        validate_allowed_thinking_levels(
            provider=ProviderName.GLM,
            model="<config-shape>",
            thinking_level=self.thinking_level,
            allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
            allow_na=False,
        )
        return self


class OpenRouterReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderName.OPENROUTER] = ProviderName.OPENROUTER
    enabled: bool | None = None
    effort: OpenRouterEffortLevel | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> OpenRouterReasoning:
        if (self.enabled is None) == (self.effort is None):
            raise ValueError(
                "openrouter reasoning requires exactly one of enabled or effort"
            )
        return self


class GoogleReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderName.GOOGLE] = ProviderName.GOOGLE
    thinking_level: ThinkingLevel = ThinkingLevel.NA
    budget_tokens: int | None = None
    include_thoughts: bool | None = None

    @field_validator("budget_tokens")
    @classmethod
    def _validate_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("google reasoning budget_tokens must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> GoogleReasoning:
        if (
            self.thinking_level == ThinkingLevel.BUDGET
            and self.budget_tokens is None
        ):
            raise ValueError("google budget thinking requires budget_tokens")
        if (
            self.thinking_level != ThinkingLevel.BUDGET
            and self.budget_tokens is not None
        ):
            raise ValueError(
                "google budget_tokens are only allowed with thinking_level='budget'"
            )
        if (
            self.include_thoughts is not None
            and self.thinking_level == ThinkingLevel.NA
        ):
            raise ValueError(
                "google include_thoughts requires an explicit thinking_level"
            )
        return self


ReasoningSpec = Annotated[
    ReasoningBudget
    | AnthropicReasoning
    | OpenAIReasoning
    | CodexReasoning
    | GlmReasoning
    | OpenRouterReasoning
    | GoogleReasoning,
    Field(discriminator="kind"),
]
REASONING_SPEC_ADAPTER = TypeAdapter(ReasoningSpec)


def parse_reasoning_spec(payload: object) -> ReasoningSpec:
    return REASONING_SPEC_ADAPTER.validate_python(payload)


# ---------------------------------------------------------------------------
# Shared validation helpers used by per-provider validators.
# ---------------------------------------------------------------------------


def raise_unsupported_thinking_level(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> Never:
    raise ValueError(
        f"thinking_level {thinking_level!r} is not supported for provider={provider!r} model={model!r}"
    )


def validate_discrete_thinking_level(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    supports_off: bool,
    supports_minimal: bool,
    supports_xhigh: bool = False,
) -> None:
    if thinking_level == ThinkingLevel.NA:
        raise ValueError(
            f"thinking_level='na' is not supported for provider={provider!r} model={model!r}"
        )
    if thinking_level == ThinkingLevel.OFF:
        if supports_off:
            return
        raise ValueError(
            f"thinking_level='off' is not supported for provider={provider!r} model={model!r}"
        )
    if thinking_level == ThinkingLevel.MINIMAL:
        if supports_minimal:
            return
        raise ValueError(
            f"thinking_level='minimal' is not supported for provider={provider!r} model={model!r}"
        )
    if thinking_level in {
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    }:
        return
    if thinking_level == ThinkingLevel.XHIGH:
        if supports_xhigh:
            return
        raise_unsupported_thinking_level(
            provider=provider,
            model=model,
            thinking_level=ThinkingLevel.XHIGH,
        )
    raise_unsupported_thinking_level(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
    )


def validate_allowed_thinking_levels(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    allowed_levels: set[ThinkingLevel],
    allow_na: bool,
) -> None:
    if thinking_level == ThinkingLevel.NA:
        if allow_na:
            return
        raise ValueError(
            f"thinking_level='na' is not supported for provider={provider!r} model={model!r}"
        )
    if thinking_level in allowed_levels:
        return
    allowed = ", ".join(str(level) for level in sorted(allowed_levels))
    raise ValueError(
        f"thinking_level {thinking_level!r} is not supported for provider={provider!r} model={model!r}; allowed levels: {allowed}"
    )


_GOOGLE_LITERAL_TO_THINKING_LEVEL: dict[str, ThinkingLevel] = {
    "minimal": ThinkingLevel.MINIMAL,
    "low": ThinkingLevel.LOW,
    "medium": ThinkingLevel.MEDIUM,
    "high": ThinkingLevel.HIGH,
}


def google_literal_to_thinking_level(level: str) -> ThinkingLevel:
    mapped = _GOOGLE_LITERAL_TO_THINKING_LEVEL.get(level)
    if mapped is None:
        expected_literals = ", ".join(
            sorted(_GOOGLE_LITERAL_TO_THINKING_LEVEL)
        )
        expected_members = ", ".join(
            e.name
            for e in sorted(
                _GOOGLE_LITERAL_TO_THINKING_LEVEL.values(),
                key=lambda x: x.value,
            )
        )
        raise ValueError(
            f"google_literal_to_thinking_level: invalid google thinking level string "
            f"{level!r}; expected one of {{{expected_literals}}} "
            f"(ThinkingLevel members: {expected_members}). "
            f"Valid strings map to ThinkingLevel; see google_literal_to_thinking_level."
        )
    return mapped


def validate_budget_range(
    *,
    provider: str,
    model: str,
    label: str,
    tokens: int,
    min_budget_tokens: int | None,
    max_budget_tokens: int | None,
) -> None:
    if min_budget_tokens is None or max_budget_tokens is None:
        raise ValueError(
            f"{label} is not supported for provider={provider!r} model={model!r}"
        )
    if tokens < min_budget_tokens or tokens > max_budget_tokens:
        raise ValueError(
            f"{label} must be between {min_budget_tokens} and {max_budget_tokens} for provider={provider!r} model={model!r}"
        )


def require_budget_tokens(
    budget_tokens: int | None,
    *,
    label: str,
    min_value: int,
) -> int:
    if budget_tokens is None:
        raise ValueError(
            f"{label} budget thinking requires budget_tokens when thinking_level is 'budget'"
        )
    if type(budget_tokens) is not int:
        raise ValueError(
            f"{label} budget_tokens must be int, got {type(budget_tokens).__name__}"
        )
    if budget_tokens < min_value:
        raise ValueError(f"{label} budget_tokens must be >= {min_value}")
    return budget_tokens


def unsupported_reasoning_kind_message(
    prefix: str, config: ReasoningSpec
) -> str:
    return f"{prefix} reasoning serializer received unsupported config kind={config.kind!r}"


def is_control_unsupported(control_mode: ControlMode | None) -> bool:
    return control_mode is None or control_mode == ControlMode.UNSUPPORTED


def dispatch_reasoning_validation(
    *,
    provider: str,
    model: str,
    reasoning: ReasoningSpec | None,
    native_spec_type: type[BaseModel],
    requires_reasoning: bool,
    validate_native: Callable[[Any], None],
    validate_top_budget: Callable[[ReasoningBudget], None],
) -> None:
    if reasoning is None:
        if requires_reasoning:
            raise ValueError(
                f"reasoning is required for provider={provider!r} model={model!r}"
            )
        return
    if isinstance(reasoning, native_spec_type):
        validate_native(reasoning)
        return
    if isinstance(reasoning, ReasoningBudget):
        validate_top_budget(reasoning)
        return
    raise ValueError(
        f"{provider} reasoning is not supported for kind={reasoning.kind!r}"
    )
