"""Reasoning configuration shared across LLM providers.

Three terms coexist in this codebase and they are NOT synonyms:

* **Reasoning** (this module) — the cross-provider abstraction. Callers
  attach a :class:`ReasoningSpec` to an :class:`~dr_llm.llm.request.LlmRequest`
  to describe how a model should think. ``ReasoningSpec`` is a discriminated
  union: either a portable :class:`ReasoningBudget` (numeric token budget) or
  one of the provider-native shapes (:class:`AnthropicReasoning`,
  :class:`OpenAIReasoning`, :class:`CodexReasoning`, :class:`GlmReasoning`,
  :class:`OpenRouterReasoning`, :class:`GoogleReasoning`).

* **Thinking level** (:class:`ThinkingLevel`) — the discrete enum used inside
  most provider-native specs. Values like ``OFF``, ``MINIMAL``, ``LOW``,
  ``MEDIUM``, ``HIGH``, ``BUDGET``, and ``ADAPTIVE`` describe *how much* the
  provider should think; the exact semantics of each level are
  provider-specific and validated per-model.

* **Effort** (``providers/effort.py`` / :class:`~dr_llm.llm.providers.effort_types.EffortSpec`)
  — a parallel system used by headless / CLI providers (claude-code,
  kimi-code, minimax, native anthropic CLI) where models advertise an effort
  tier instead of a thinking level.

A given model supports **either** an effort axis **or** a reasoning axis,
never both. ``LlmRequest`` exposes both fields (``effort`` and ``reasoning``)
and the per-provider validators reject the combination that does not apply.

The validation helpers in this module are shared across providers that use
the discrete-tier convention (OpenAI, Codex, OpenRouter), so they are named
generically (e.g. :func:`validate_discrete_thinking_level`) rather than
after a single provider.
"""

from __future__ import annotations

from collections.abc import Callable
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

from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.openrouter.policy import OpenRouterEffortLevel
from dr_llm.llm.providers.reasoning_capability_types import ReasoningCapabilities


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


class ThinkingLevel(StrEnum):
    NA = "na"
    OFF = "off"
    BUDGET = "budget"
    ADAPTIVE = "adaptive"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


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
        if self.thinking_level == ThinkingLevel.BUDGET and self.budget_tokens is None:
            raise ValueError("anthropic budget thinking requires budget_tokens")
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

    kind: Literal["openai"] = "openai"
    thinking_level: ThinkingLevel = ThinkingLevel.NA


class CodexReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["codex"] = "codex"
    thinking_level: ThinkingLevel = ThinkingLevel.NA


class GlmReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["glm"] = "glm"
    thinking_level: ThinkingLevel = ThinkingLevel.NA

    @model_validator(mode="after")
    def _validate_shape(self) -> GlmReasoning:
        validate_allowed_thinking_levels(
            provider="glm",
            model="<config-shape>",
            thinking_level=self.thinking_level,
            allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
            allow_na=False,
        )
        return self


class OpenRouterReasoning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["openrouter"] = "openrouter"
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

    kind: Literal["google"] = "google"
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
        if self.thinking_level == ThinkingLevel.BUDGET and self.budget_tokens is None:
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


def validate_discrete_thinking_level(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    supports_off: bool,
    supports_minimal: bool,
    supports_xhigh: bool = False,
) -> None:
    """Validate a discrete (OFF/MINIMAL/LOW/MEDIUM/HIGH/XHIGH) thinking level.

    Used by every provider whose API exposes thinking as a fixed set of
    tiers rather than a numeric token budget — currently OpenAI (Responses
    API), Codex (CLI), and OpenRouter when the upstream model uses the
    OpenAI-style ``reasoning.effort`` shape. ``OFF`` and ``MINIMAL`` are
    optional per-model; ``XHIGH`` is opt-in per provider/model family;
    ``LOW``/``MEDIUM``/``HIGH`` are always accepted when this validator is
    reached.
    """
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
        raise ValueError(
            f"thinking_level='xhigh' is not supported for provider={provider!r} model={model!r}"
        )
    raise ValueError(
        f"thinking_level {thinking_level!r} is not supported for provider={provider!r} model={model!r}"
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
    allowed = ", ".join(level.value for level in sorted(allowed_levels))
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
        expected_literals = ", ".join(sorted(_GOOGLE_LITERAL_TO_THINKING_LEVEL))
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


def require_budget_tokens(
    budget_tokens: int | None,
    *,
    label: str,
    min_value: int,
) -> int:
    """Validate budget_tokens for serializer payloads (strictly typed int >= min_value)."""
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


class BaseProviderReasoningConfig(BaseModel):
    """Common Pydantic boilerplate for per-provider reasoning serializer configs."""

    model_config = ConfigDict(frozen=True)

    warnings: list[ReasoningWarning] = Field(default_factory=list)


def unsupported_reasoning_kind_message(prefix: str, config: ReasoningSpec) -> str:
    return f"{prefix} reasoning serializer received unsupported config kind={config.kind!r}"


def is_reasoning_unsupported(capabilities: ReasoningCapabilities | None) -> bool:
    return capabilities is None or capabilities.mode == "unsupported"


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
    """Shared dispatch for per-provider reasoning validators.

    Implements the common skeleton: ``None`` → check requirement; native spec →
    delegate to ``validate_native``; ``ReasoningBudget`` → delegate to
    ``validate_top_budget``; otherwise raise an unsupported-kind error.
    """
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
