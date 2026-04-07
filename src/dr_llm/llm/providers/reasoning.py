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

from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
    ANTHROPIC_BUDGET_THINKING_SUPPORTED,
    ANTHROPIC_THINKING_MAX_BUDGET_TOKENS,
    ANTHROPIC_THINKING_MIN_BUDGET_TOKENS,
)
from dr_llm.llm.providers.headless.codex_thinking import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
)
from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterEffortLevel,
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.providers.reasoning_capabilities import (
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


class ThinkingLevel(StrEnum):
    NA = "na"
    OFF = "off"
    BUDGET = "budget"
    ADAPTIVE = "adaptive"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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
        _validate_allowed_thinking_levels(
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


def validate_reasoning(
    *,
    provider: str,
    model: str,
    reasoning: ReasoningSpec | None,
) -> None:
    if provider == "openrouter" and openrouter_model_policy(model) is None:
        raise ValueError(f"openrouter model={model!r} is not in the curated allowlist")
    capabilities = reasoning_capabilities_for_model(provider=provider, model=model)
    if reasoning is None:
        if _requires_explicit_reasoning(
            provider=provider,
            model=model,
            capabilities=capabilities,
        ):
            raise ValueError(
                f"reasoning is required for provider={provider!r} model={model!r}"
            )
        return
    if isinstance(reasoning, AnthropicReasoning):
        _validate_anthropic_reasoning(
            provider=provider,
            model=model,
            thinking_level=reasoning.thinking_level,
            budget_tokens=reasoning.budget_tokens,
            display=reasoning.display,
        )
        return
    if isinstance(reasoning, OpenAIReasoning):
        _validate_openai_reasoning(
            provider=provider,
            model=model,
            thinking_level=reasoning.thinking_level,
        )
        return
    if isinstance(reasoning, CodexReasoning):
        _validate_codex_reasoning(
            provider=provider,
            model=model,
            thinking_level=reasoning.thinking_level,
        )
        return
    if isinstance(reasoning, GlmReasoning):
        _validate_glm_reasoning(
            provider=provider,
            model=model,
            thinking_level=reasoning.thinking_level,
        )
        return
    if isinstance(reasoning, GoogleReasoning):
        _validate_google_reasoning(
            provider=provider,
            model=model,
            thinking_level=reasoning.thinking_level,
            budget_tokens=reasoning.budget_tokens,
        )
        return
    if isinstance(reasoning, OpenRouterReasoning):
        _validate_openrouter_reasoning(
            provider=provider,
            model=model,
            enabled=reasoning.enabled,
            effort=reasoning.effort,
        )
        return
    if provider == "anthropic" and isinstance(reasoning, ReasoningBudget):
        if model not in ANTHROPIC_BUDGET_THINKING_SUPPORTED:
            raise ValueError(
                f"anthropic budget thinking is not supported for model={model!r}"
            )
        _validate_anthropic_budget_range(
            provider=provider,
            model=model,
            label="anthropic budget_tokens",
            tokens=reasoning.tokens,
        )
        return
    if provider == "claude-code" and isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"claude-code does not support budget thinking for model={model!r}"
        )
    if provider == "kimi-code" and isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "kimi-code requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='budget', budget_tokens=...)"
        )
    if provider == "minimax" and isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "minimax requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='na')"
        )
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
    match reasoning:
        case AnthropicReasoning(
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            display=display,
        ):
            _validate_anthropic_reasoning(
                provider=provider,
                model=model,
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            )
        case OpenAIReasoning(thinking_level=thinking_level):
            _validate_openai_reasoning(
                provider=provider,
                model=model,
                thinking_level=thinking_level,
            )
        case CodexReasoning(thinking_level=thinking_level):
            _validate_codex_reasoning(
                provider=provider,
                model=model,
                thinking_level=thinking_level,
            )
        case GlmReasoning(thinking_level=thinking_level):
            _validate_glm_reasoning(
                provider=provider,
                model=model,
                thinking_level=thinking_level,
            )
        case ReasoningBudget(tokens=tokens):
            if provider == "anthropic":
                if model not in ANTHROPIC_BUDGET_THINKING_SUPPORTED:
                    raise ValueError(
                        f"anthropic budget thinking is not supported for model={model!r}"
                    )
                _validate_anthropic_budget_range(
                    provider=provider,
                    model=model,
                    label="anthropic budget_tokens",
                    tokens=tokens,
                )
                return
            if provider == "claude-code":
                raise ValueError(
                    f"claude-code does not support budget thinking for model={model!r}"
                )
            if capabilities.mode == "unsupported":
                raise ValueError(
                    f"Reasoning is not supported for provider={provider!r} model={model!r}"
                )
            _validate_budget_range(
                provider=provider,
                model=model,
                label="reasoning budget",
                tokens=tokens,
                capabilities=capabilities,
            )
        case GoogleReasoning(
            thinking_level=thinking_level, budget_tokens=budget_tokens
        ):
            _validate_google_reasoning(
                provider=provider,
                model=model,
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
            )
        case OpenRouterReasoning(enabled=enabled, effort=effort):
            _validate_openrouter_reasoning(
                provider=provider,
                model=model,
                enabled=enabled,
                effort=effort,
            )
        case _:
            raise ValueError(
                f"Unsupported reasoning configuration for provider={provider!r} model={model!r}"
            )


def _validate_anthropic_reasoning(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
    display: Literal["summarized", "omitted"] | None,
) -> None:
    if provider == "anthropic":
        if thinking_level == ThinkingLevel.NA:
            if model in (
                set(ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED)
                | set(ANTHROPIC_BUDGET_THINKING_SUPPORTED)
            ):
                raise ValueError(
                    f"thinking_level='na' is not supported for provider={provider!r} model={model!r}"
                )
            return
        if thinking_level == ThinkingLevel.OFF:
            return
        if thinking_level == ThinkingLevel.ADAPTIVE:
            if model not in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
                raise ValueError(
                    f"anthropic adaptive thinking is not supported for model={model!r}"
                )
            return
        if thinking_level == ThinkingLevel.BUDGET:
            if model not in ANTHROPIC_BUDGET_THINKING_SUPPORTED:
                raise ValueError(
                    f"anthropic budget thinking is not supported for model={model!r}"
                )
            if budget_tokens is None:
                raise ValueError(
                    "anthropic budget thinking requires budget_tokens when "
                    "thinking_level is 'budget'"
                )
            if type(budget_tokens) is not int:
                raise ValueError(
                    "anthropic budget_tokens must be int, got "
                    f"{type(budget_tokens).__name__}"
                )
            if budget_tokens <= 0:
                raise ValueError("anthropic budget_tokens must be > 0")
            _validate_anthropic_budget_range(
                provider=provider,
                model=model,
                label="anthropic budget_tokens",
                tokens=budget_tokens,
            )
            return
        raise ValueError(
            f"Unsupported anthropic thinking level {thinking_level!r} for model={model!r}"
        )

    if provider == "claude-code":
        if display is not None:
            raise ValueError("claude-code does not support anthropic display controls")
        if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
            if thinking_level != ThinkingLevel.ADAPTIVE:
                raise ValueError(
                    f"claude-code model {model!r} only supports anthropic thinking_level='adaptive'"
                )
            return
        if thinking_level != ThinkingLevel.NA:
            raise ValueError(
                f"claude-code model {model!r} does not support explicit anthropic thinking; use thinking_level='na'"
            )
        return

    if provider == "minimax":
        if display is not None:
            raise ValueError(f"{provider} does not support anthropic display controls")
        if budget_tokens is not None:
            raise ValueError(
                f"{provider} does not support anthropic budget thinking controls"
            )
        if thinking_level != ThinkingLevel.NA:
            raise ValueError(
                f"{provider} does not support explicit anthropic thinking; use thinking_level='na'"
            )
        return

    if provider == "kimi-code":
        if display is not None:
            raise ValueError("kimi-code does not support anthropic display controls")
        if thinking_level == ThinkingLevel.NA:
            if budget_tokens is not None:
                raise ValueError(
                    "kimi-code budget_tokens require thinking_level='budget'"
                )
            return
        if thinking_level == ThinkingLevel.OFF:
            if budget_tokens is not None:
                raise ValueError(
                    "kimi-code budget_tokens require thinking_level='budget'"
                )
            return
        if thinking_level == ThinkingLevel.ADAPTIVE:
            if budget_tokens is not None:
                raise ValueError(
                    "kimi-code budget_tokens require thinking_level='budget'"
                )
            return
        if thinking_level == ThinkingLevel.BUDGET:
            if budget_tokens is None:
                raise ValueError("kimi-code budget thinking requires budget_tokens")
            _validate_anthropic_budget_range(
                provider=provider,
                model=model,
                label="kimi-code budget_tokens",
                tokens=budget_tokens,
            )
            return
        raise ValueError(
            f"Unsupported kimi-code thinking level {thinking_level!r} for model={model!r}"
        )

    raise ValueError(f"anthropic reasoning is not supported for provider={provider!r}")


def _validate_openai_reasoning(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> None:
    if provider not in {"openai", "openrouter"}:
        raise ValueError(f"openai thinking is not supported for provider={provider!r}")
    if not openai_supports_configurable_thinking(model):
        raise ValueError(f"openai thinking is not supported for model={model!r}")
    _validate_openai_style_thinking_level(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
        supports_off=openai_supports_off_thinking(model),
        supports_minimal=openai_supports_minimal_thinking(model),
    )


def _validate_openrouter_reasoning(
    *,
    provider: str,
    model: str,
    enabled: bool | None,
    effort: OpenRouterEffortLevel | None,
) -> None:
    if provider != "openrouter":
        raise ValueError(
            f"openrouter reasoning is not supported for provider={provider!r}"
        )
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
        assert enabled is not None
        if not enabled and not policy.supports_disable:
            raise ValueError(
                f"openrouter reasoning cannot be disabled for model={model!r}"
            )
        return
    if enabled is not None:
        raise ValueError(
            f"openrouter enabled controls are not supported for model={model!r}"
        )
    assert effort is not None
    if effort not in policy.allowed_efforts:
        allowed = ", ".join(policy.allowed_efforts)
        raise ValueError(
            f"openrouter effort={effort!r} is not supported for model={model!r}; allowed levels: {allowed}"
        )


def _validate_codex_reasoning(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> None:
    if provider != "codex":
        raise ValueError(f"codex thinking is not supported for provider={provider!r}")
    if not codex_supports_configurable_thinking(model):
        raise ValueError(f"codex thinking is not supported for model={model!r}")
    _validate_openai_style_thinking_level(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
        supports_off=codex_supports_off_thinking(model),
        supports_minimal=codex_supports_minimal_thinking(model),
    )


def _validate_glm_reasoning(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> None:
    if provider != "glm":
        raise ValueError(f"glm thinking is not supported for provider={provider!r}")
    _validate_allowed_thinking_levels(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
        allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
        allow_na=False,
    )


def _validate_google_reasoning(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> None:
    if provider != "google":
        raise ValueError(f"google thinking is not supported for provider={provider!r}")
    capabilities = reasoning_capabilities_for_model(provider=provider, model=model)
    if capabilities is None or capabilities.mode == "unsupported":
        if thinking_level == ThinkingLevel.NA:
            return
        raise ValueError(f"google thinking is not supported for model={model!r}")
    if thinking_level == ThinkingLevel.NA:
        raise ValueError(
            f"thinking_level='na' is not supported for provider={provider!r} model={model!r}"
        )
    if capabilities.mode == "google_budget":
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
            if type(budget_tokens) is not int:
                raise ValueError(
                    "google budget_tokens must be int, got "
                    f"{type(budget_tokens).__name__}"
                )
            if budget_tokens < 0:
                raise ValueError("google budget_tokens must be >= 0")
            _validate_budget_range(
                provider=provider,
                model=model,
                label="google thinking_budget",
                tokens=budget_tokens,
                capabilities=capabilities,
            )
            return
        raise ValueError(
            f"google model {model!r} does not support thinking_level={thinking_level!r}; use off, adaptive, or budget"
        )
    if capabilities.mode == "google_level":
        allowed_levels = {
            _google_literal_to_thinking_level(level)
            for level in capabilities.google_levels
        }
        _validate_allowed_thinking_levels(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            allowed_levels=allowed_levels,
            allow_na=False,
        )
        return
    raise ValueError(
        f"google reasoning is not supported for provider={provider!r} model={model!r}"
    )


def _validate_openai_style_thinking_level(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    supports_off: bool,
    supports_minimal: bool,
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
    raise ValueError(
        f"thinking_level {thinking_level!r} is not supported for provider={provider!r} model={model!r}"
    )


def _validate_allowed_thinking_levels(
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


def _google_literal_to_thinking_level(level: str) -> ThinkingLevel:
    return {
        "minimal": ThinkingLevel.MINIMAL,
        "low": ThinkingLevel.LOW,
        "medium": ThinkingLevel.MEDIUM,
        "high": ThinkingLevel.HIGH,
    }[level]


def _requires_explicit_reasoning(
    *,
    model: str,
    provider: str,
    capabilities: ReasoningCapabilities | None,
) -> bool:
    if capabilities is None or capabilities.mode == "unsupported":
        return False
    if provider in {
        "openai",
        "openrouter",
        "codex",
        "glm",
        "google",
        "anthropic",
        "minimax",
    }:
        return True
    if provider == "claude-code":
        return model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED
    return False


def _validate_anthropic_budget_range(
    *,
    provider: str,
    model: str,
    label: str,
    tokens: int,
) -> None:
    if (
        tokens < ANTHROPIC_THINKING_MIN_BUDGET_TOKENS
        or tokens > ANTHROPIC_THINKING_MAX_BUDGET_TOKENS
    ):
        raise ValueError(
            f"{label} must be between {ANTHROPIC_THINKING_MIN_BUDGET_TOKENS} and {ANTHROPIC_THINKING_MAX_BUDGET_TOKENS} for provider={provider!r} model={model!r}"
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
