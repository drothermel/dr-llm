from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ProviderName, ThinkingLevel
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
)
from dr_llm.llm.providers.effort import supported_effort_levels
from dr_llm.llm.providers.headless.codex.capabilities import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningSpec,
    google_literal_to_thinking_level,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    reasoning_capabilities_for_model,
)


class ReasoningControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    supported_thinking_levels: tuple[ThinkingLevel, ...]
    default_thinking_level: ThinkingLevel
    supported_effort_levels: tuple[EffortSpec, ...]
    default_effort: EffortSpec
    default_reasoning: ReasoningSpec | None


def reasoning_controls_for_model(
    *, provider: str, model: str
) -> ReasoningControls:
    return ReasoningControls(
        provider=provider,
        model=model,
        supported_thinking_levels=supported_thinking_levels(
            provider=provider, model=model
        ),
        default_thinking_level=default_thinking_level(
            provider=provider, model=model
        ),
        supported_effort_levels=supported_effort_levels(
            provider=provider, model=model
        ),
        default_effort=default_effort(provider=provider, model=model),
        default_reasoning=default_reasoning(provider=provider, model=model),
    )


def supported_thinking_levels(
    *, provider: str, model: str
) -> tuple[ThinkingLevel, ...]:
    def get_capabilities() -> ReasoningCapabilities | None:
        return reasoning_capabilities_for_model(provider=provider, model=model)

    def supported_kimi_code_thinking_levels() -> tuple[ThinkingLevel, ...]:
        capabilities = get_capabilities()
        if _is_reasoning_unsupported(capabilities):
            return (ThinkingLevel.NA,)
        return (
            ThinkingLevel.OFF,
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.BUDGET,
        )

    dispatch: dict[str, Callable[[], tuple[ThinkingLevel, ...]]] = {
        ProviderName.OPENAI: lambda: _supported_openai_thinking_levels(model),
        ProviderName.CODEX: lambda: _supported_codex_thinking_levels(model),
        ProviderName.CLAUDE_CODE: lambda: (
            _supported_claude_code_thinking_levels(model)
        ),
        ProviderName.MINIMAX: lambda: (ThinkingLevel.NA,),
        ProviderName.KIMI_CODE: supported_kimi_code_thinking_levels,
        ProviderName.OPENROUTER: lambda: (ThinkingLevel.NA,),
    }
    return dispatch.get(
        provider,
        lambda: _supported_capability_thinking_levels(
            provider=provider,
            model=model,
            capabilities=get_capabilities(),
        ),
    )()


def default_thinking_level(*, provider: str, model: str) -> ThinkingLevel:
    levels = supported_thinking_levels(provider=provider, model=model)
    for level in (
        ThinkingLevel.OFF,
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    ):
        if level in levels:
            return level
    return ThinkingLevel.NA


def default_effort(*, provider: str, model: str) -> EffortSpec:
    levels = supported_effort_levels(provider=provider, model=model)
    if levels:
        return levels[0]
    return EffortSpec.NA


def default_reasoning(*, provider: str, model: str) -> ReasoningSpec | None:
    if provider == ProviderName.OPENROUTER:
        return _default_openrouter_reasoning(model)
    thinking_level = default_thinking_level(provider=provider, model=model)
    capabilities = reasoning_capabilities_for_model(
        provider=provider, model=model
    )
    budget_tokens = _default_budget_tokens(capabilities)
    return reasoning_for_thinking_level(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
        budget_tokens=budget_tokens,
        explicit_na=provider == ProviderName.MINIMAX,
    )


def reasoning_for_thinking_level(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None = None,
    explicit_na: bool = False,
) -> ReasoningSpec | None:
    if provider == ProviderName.OPENAI:
        return _reasoning_for_openai(thinking_level)
    if provider == ProviderName.CODEX:
        return _reasoning_for_codex(thinking_level)
    if provider == ProviderName.GOOGLE:
        return _reasoning_for_google(
            thinking_level=thinking_level, budget_tokens=budget_tokens
        )
    if provider == ProviderName.GLM:
        return _reasoning_for_glm(thinking_level)
    if provider in {ProviderName.ANTHROPIC, ProviderName.KIMI_CODE}:
        return _reasoning_for_anthropic_style(
            provider=provider,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            explicit_na=explicit_na,
        )
    if provider in {ProviderName.CLAUDE_CODE, ProviderName.MINIMAX}:
        return _reasoning_for_headless_anthropic_style(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            explicit_na=explicit_na,
        )
    if provider == ProviderName.OPENROUTER:
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(
            f"openrouter does not support thinking_level={thinking_level!r}"
        )
    raise ValueError(f"unsupported provider: {provider!r}")


def _supported_openai_thinking_levels(
    model: str,
) -> tuple[ThinkingLevel, ...]:
    return _supported_openai_style_thinking_levels(
        supports_configurable=openai_supports_configurable_thinking(model),
        supports_off=openai_supports_off_thinking(model),
        supports_minimal=openai_supports_minimal_thinking(model),
        supports_xhigh=False,
    )


def _supported_codex_thinking_levels(model: str) -> tuple[ThinkingLevel, ...]:
    return _supported_openai_style_thinking_levels(
        supports_configurable=codex_supports_configurable_thinking(model),
        supports_off=codex_supports_off_thinking(model),
        supports_minimal=codex_supports_minimal_thinking(model),
        supports_xhigh=codex_supports_configurable_thinking(model),
    )


def _supported_openai_style_thinking_levels(
    *,
    supports_configurable: bool,
    supports_off: bool,
    supports_minimal: bool,
    supports_xhigh: bool,
) -> tuple[ThinkingLevel, ...]:
    if not supports_configurable:
        return (ThinkingLevel.NA,)
    levels: list[ThinkingLevel] = []
    if supports_off:
        levels.append(ThinkingLevel.OFF)
    elif supports_minimal:
        levels.append(ThinkingLevel.MINIMAL)
    levels.extend(
        [
            ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH,
        ]
    )
    if supports_xhigh:
        levels.append(ThinkingLevel.XHIGH)
    return tuple(levels)


def _supported_claude_code_thinking_levels(
    model: str,
) -> tuple[ThinkingLevel, ...]:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return (ThinkingLevel.ADAPTIVE,)
    return (ThinkingLevel.NA,)


def _supported_capability_thinking_levels(
    *,
    provider: str,
    model: str,
    capabilities: ReasoningCapabilities | None,
) -> tuple[ThinkingLevel, ...]:
    if _is_reasoning_unsupported(capabilities):
        return (ThinkingLevel.NA,)
    assert capabilities is not None
    if capabilities.mode == "google_budget":
        return (
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.OFF,
            ThinkingLevel.BUDGET,
        )
    if capabilities.mode == "google_level":
        return tuple(
            google_literal_to_thinking_level(level)
            for level in capabilities.google_levels
        )
    if capabilities.mode == ProviderName.GLM:
        return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
    if capabilities.mode == "anthropic_budget":
        return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
    if capabilities.mode == "anthropic_effort":
        return _supported_anthropic_effort_thinking_levels(model)
    if capabilities.mode == "anthropic_effort_and_budget":
        return (
            *_supported_anthropic_effort_thinking_levels(model),
            ThinkingLevel.BUDGET,
        )
    raise ValueError(
        f"unexpected reasoning mode for provider={provider!r} "
        f"model={model!r}: {capabilities.mode!r}"
    )


def _supported_anthropic_effort_thinking_levels(
    model: str,
) -> tuple[ThinkingLevel, ...]:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
    return (ThinkingLevel.OFF,)


def _default_openrouter_reasoning(model: str) -> OpenRouterReasoning | None:
    policy = openrouter_model_policy(model)
    if policy is None:
        raise ValueError(f"missing openrouter policy for model={model!r}")
    if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
        return None
    if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        # Default to disabled when the model supports it; otherwise request enabled.
        if policy.supports_disable:
            return OpenRouterReasoning(enabled=False)
        return OpenRouterReasoning(enabled=True)
    return OpenRouterReasoning(effort=policy.allowed_efforts[0])


def _reasoning_for_openai(
    thinking_level: ThinkingLevel,
) -> OpenAIReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        return None
    return OpenAIReasoning(thinking_level=thinking_level)


def _reasoning_for_codex(
    thinking_level: ThinkingLevel,
) -> CodexReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        return None
    return CodexReasoning(thinking_level=thinking_level)


def _reasoning_for_google(
    *,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> GoogleReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        return None
    if thinking_level == ThinkingLevel.BUDGET:
        return GoogleReasoning(
            thinking_level=thinking_level,
            budget_tokens=_require_budget_tokens(
                provider=ProviderName.GOOGLE, budget_tokens=budget_tokens
            ),
        )
    return GoogleReasoning(thinking_level=thinking_level)


def _reasoning_for_glm(thinking_level: ThinkingLevel) -> GlmReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        return None
    return GlmReasoning(thinking_level=thinking_level)


def _reasoning_for_anthropic_style(
    *,
    provider: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
    explicit_na: bool,
) -> AnthropicReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        if explicit_na:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        return None
    if thinking_level == ThinkingLevel.BUDGET:
        return AnthropicReasoning(
            thinking_level=thinking_level,
            budget_tokens=_require_budget_tokens(
                provider=provider, budget_tokens=budget_tokens
            ),
        )
    return AnthropicReasoning(thinking_level=thinking_level)


def _reasoning_for_headless_anthropic_style(
    *,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    explicit_na: bool,
) -> AnthropicReasoning | None:
    if thinking_level == ThinkingLevel.ADAPTIVE:
        return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    if thinking_level == ThinkingLevel.NA:
        if explicit_na:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        return None
    raise ValueError(
        f"unsupported {provider} thinking level for model={model!r}: "
        f"{thinking_level!r}"
    )


def _default_budget_tokens(
    capabilities: ReasoningCapabilities | None,
) -> int | None:
    if capabilities is None:
        return None
    return capabilities.min_budget_tokens


def _require_budget_tokens(*, provider: str, budget_tokens: int | None) -> int:
    if budget_tokens is None:
        raise ValueError(f"{provider} budget thinking requires budget_tokens")
    return budget_tokens


def _is_reasoning_unsupported(
    capabilities: ReasoningCapabilities | None,
) -> bool:
    return capabilities is None or capabilities.mode == "unsupported"
