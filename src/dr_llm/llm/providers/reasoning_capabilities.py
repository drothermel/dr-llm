from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.providers.headless.codex_thinking import (
    codex_supports_configurable_thinking,
)
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
)

GoogleThinkingLevel = Literal["minimal", "low", "medium", "high"]
ReasoningMode = Literal[
    "unsupported",
    "openai_effort",
    "openrouter_toggle",
    "openrouter_effort",
    "glm",
    "google_budget",
    "google_level",
    "anthropic_budget",
    "anthropic_effort",
    "anthropic_effort_and_budget",
    "claude_cli_effort",
    "codex_cli_effort",
    "kimi_code_effort_and_budget",
    "minimax_effort",
]


class ReasoningCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ReasoningMode = "unsupported"
    google_levels: tuple[GoogleThinkingLevel, ...] = ()
    min_budget_tokens: int | None = None
    max_budget_tokens: int | None = None
    supports_dynamic: bool = False

    @property
    def supports_reasoning(self) -> bool:
        return self.mode != "unsupported"


class ReasoningCapabilityRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    exact_model: str | None = None
    model_prefix: str | None = None
    capabilities: ReasoningCapabilities

    @model_validator(mode="after")
    def _validate_shape(self) -> ReasoningCapabilityRule:
        if (self.exact_model is None) == (self.model_prefix is None):
            raise ValueError(
                "reasoning capability rules must define exactly one of exact_model or model_prefix"
            )
        return self

    def matches(self, *, provider: str, model: str) -> bool:
        if provider != self.provider:
            return False
        if self.exact_model is not None:
            return model == self.exact_model
        assert self.model_prefix is not None
        return model.startswith(self.model_prefix)


_GOOGLE_25_FLASH_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=1,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_FLASH_LITE_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=512,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_PRO_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=128,
    max_budget_tokens=32768,
    supports_dynamic=True,
)
_GOOGLE_3_CAPS = ReasoningCapabilities(
    mode="google_level",
    google_levels=("minimal", "low", "medium", "high"),
)
_GEMMA_4_CAPS = ReasoningCapabilities(
    mode="google_level",
    google_levels=("minimal", "high"),
)
_ANTHROPIC_BUDGET_CAPS = ReasoningCapabilities(
    mode="anthropic_budget",
    min_budget_tokens=1024,
    max_budget_tokens=128000,
)
_ANTHROPIC_SONNET_46_CAPS = ReasoningCapabilities(mode="anthropic_effort")
_ANTHROPIC_OPUS_45_CAPS = ReasoningCapabilities(
    mode="anthropic_effort_and_budget",
    min_budget_tokens=1024,
    max_budget_tokens=128000,
)
_ANTHROPIC_OPUS_46_CAPS = ReasoningCapabilities(mode="anthropic_effort")
_CLAUDE_HEADLESS_CAPS = ReasoningCapabilities(mode="claude_cli_effort")
_CODEX_CLI_EFFORT_CAPS = ReasoningCapabilities(mode="codex_cli_effort")
_GLM_THINKING_CAPS = ReasoningCapabilities(mode="glm")
_KIMI_CODE_CAPS = ReasoningCapabilities(
    mode="kimi_code_effort_and_budget",
    min_budget_tokens=1024,
    max_budget_tokens=128000,
)
_MINIMAX_CAPS = ReasoningCapabilities(mode="minimax_effort")

CURATED_REASONING_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        provider="google",
        model_prefix="gemini-2.5-flash-lite-preview",
        capabilities=_GOOGLE_25_FLASH_LITE_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="google",
        model_prefix="gemini-2.5-flash-lite",
        capabilities=_GOOGLE_25_FLASH_LITE_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="google",
        model_prefix="gemini-2.5-flash-preview",
        capabilities=_GOOGLE_25_FLASH_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="google",
        model_prefix="gemini-2.5-flash",
        capabilities=_GOOGLE_25_FLASH_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="google",
        model_prefix="gemini-2.5-pro",
        capabilities=_GOOGLE_25_PRO_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="google", model_prefix="gemini-3", capabilities=_GOOGLE_3_CAPS
    ),
    ReasoningCapabilityRule(
        provider="google", model_prefix="gemma-4", capabilities=_GEMMA_4_CAPS
    ),
    ReasoningCapabilityRule(
        provider="glm",
        model_prefix="glm-5",
        capabilities=_GLM_THINKING_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="glm",
        model_prefix="glm-4.7",
        capabilities=_GLM_THINKING_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="glm",
        model_prefix="glm-4.6",
        capabilities=_GLM_THINKING_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="glm",
        model_prefix="glm-4.5",
        capabilities=_GLM_THINKING_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-opus-4-6",
        capabilities=_ANTHROPIC_OPUS_46_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-sonnet-4-6",
        capabilities=_ANTHROPIC_SONNET_46_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-opus-4-5",
        capabilities=_ANTHROPIC_OPUS_45_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-opus-4-1",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-opus-4-",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-sonnet-4-5",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-sonnet-4-",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-3-7-sonnet",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="anthropic",
        model_prefix="claude-haiku-4-5",
        capabilities=_ANTHROPIC_BUDGET_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="claude-code",
        model_prefix="claude-",
        capabilities=_CLAUDE_HEADLESS_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="kimi-code",
        exact_model="kimi-for-coding",
        capabilities=_KIMI_CODE_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="minimax",
        model_prefix="MiniMax-",
        capabilities=_MINIMAX_CAPS,
    ),
)


def reasoning_capabilities_for_model(
    *,
    provider: str,
    model: str,
) -> ReasoningCapabilities | None:
    if provider == "openrouter":
        policy = openrouter_model_policy(model)
        if policy is None:
            return None
        if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
            return ReasoningCapabilities(mode="openrouter_toggle")
        if policy.request_style == OpenRouterReasoningRequestStyle.EFFORT:
            return ReasoningCapabilities(mode="openrouter_effort")
        return ReasoningCapabilities(mode="unsupported")
    if provider == "openai" and openai_supports_configurable_thinking(model):
        return ReasoningCapabilities(mode="openai_effort")
    if provider == "codex" and codex_supports_configurable_thinking(model):
        return _CODEX_CLI_EFFORT_CAPS
    exact_rules = [
        rule for rule in CURATED_REASONING_CAPABILITY_RULES if rule.exact_model
    ]
    for rule in exact_rules:
        if rule.matches(provider=provider, model=model):
            return rule.capabilities

    prefix_rules = sorted(
        (
            rule
            for rule in CURATED_REASONING_CAPABILITY_RULES
            if rule.model_prefix is not None
        ),
        key=lambda rule: len(str(rule.model_prefix)),
        reverse=True,
    )
    for rule in prefix_rules:
        if rule.matches(provider=provider, model=model):
            return rule.capabilities
    return None
