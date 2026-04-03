from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

GenericEffortLevel = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
GoogleThinkingLevel = Literal["minimal", "low", "medium", "high"]
AnthropicEffortLevel = Literal["low", "medium", "high", "max"]
ReasoningMode = Literal[
    "unsupported",
    "openai_effort",
    "google_budget",
    "google_level",
    "anthropic_budget",
    "anthropic_effort",
    "anthropic_effort_and_budget",
    "claude_cli_effort",
    "codex_headless",
]


class ReasoningCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ReasoningMode = "unsupported"
    generic_effort_levels: tuple[GenericEffortLevel, ...] = ()
    google_levels: tuple[GoogleThinkingLevel, ...] = ()
    anthropic_effort_levels: tuple[AnthropicEffortLevel, ...] = ()
    min_budget_tokens: int | None = None
    max_budget_tokens: int | None = None
    supports_off: bool = False
    supports_dynamic: bool = False
    supports_display: bool = False
    supports_adaptive: bool = False
    notes: tuple[str, ...] = ()

    @property
    def supports_reasoning(self) -> bool:
        return self.mode not in {"unsupported", "codex_headless"}


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


_OPENAI_CAPS_GPT5 = ReasoningCapabilities(
    mode="openai_effort",
    generic_effort_levels=("minimal", "low", "medium", "high"),
    supports_off=False,
)
_OPENAI_CAPS_GPT5_1 = ReasoningCapabilities(
    mode="openai_effort",
    generic_effort_levels=("none", "low", "medium", "high"),
    supports_off=True,
)
_OPENAI_CAPS_GPT5_2_PLUS = ReasoningCapabilities(
    mode="openai_effort",
    generic_effort_levels=("none", "low", "medium", "high", "xhigh"),
    supports_off=True,
)
_GOOGLE_25_FLASH_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=1,
    max_budget_tokens=24576,
    supports_off=True,
    supports_dynamic=True,
)
_GOOGLE_25_FLASH_LITE_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=512,
    max_budget_tokens=24576,
    supports_off=True,
    supports_dynamic=True,
)
_GOOGLE_25_PRO_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=128,
    max_budget_tokens=32768,
    supports_off=False,
    supports_dynamic=True,
)
_GOOGLE_3_CAPS = ReasoningCapabilities(
    mode="google_level",
    generic_effort_levels=("low", "medium", "high"),
    google_levels=("minimal", "low", "medium", "high"),
)
_ANTHROPIC_BUDGET_CAPS = ReasoningCapabilities(
    mode="anthropic_budget",
    min_budget_tokens=1024,
    max_budget_tokens=128000,
    supports_display=True,
)
_ANTHROPIC_SONNET_46_CAPS = ReasoningCapabilities(
    mode="anthropic_effort",
    generic_effort_levels=("low", "medium", "high"),
    anthropic_effort_levels=("low", "medium", "high"),
    supports_display=True,
    supports_adaptive=True,
)
_ANTHROPIC_OPUS_45_CAPS = ReasoningCapabilities(
    mode="anthropic_effort_and_budget",
    generic_effort_levels=("low", "medium", "high"),
    anthropic_effort_levels=("low", "medium", "high"),
    min_budget_tokens=1024,
    max_budget_tokens=128000,
    supports_display=True,
)
_ANTHROPIC_OPUS_46_CAPS = ReasoningCapabilities(
    mode="anthropic_effort",
    generic_effort_levels=("low", "medium", "high", "max"),
    anthropic_effort_levels=("low", "medium", "high", "max"),
    supports_display=True,
    supports_adaptive=True,
)
_CLAUDE_HEADLESS_CAPS = ReasoningCapabilities(
    mode="claude_cli_effort",
    generic_effort_levels=("low", "medium", "high", "max"),
)
_CODEX_HEADLESS_CAPS = ReasoningCapabilities(mode="codex_headless")

CURATED_REASONING_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        provider="openai", exact_model="gpt-5", capabilities=_OPENAI_CAPS_GPT5
    ),
    ReasoningCapabilityRule(
        provider="openai", exact_model="gpt-5-mini", capabilities=_OPENAI_CAPS_GPT5
    ),
    ReasoningCapabilityRule(
        provider="openai", exact_model="gpt-5.1", capabilities=_OPENAI_CAPS_GPT5_1
    ),
    ReasoningCapabilityRule(
        provider="openai", model_prefix="gpt-5.1-", capabilities=_OPENAI_CAPS_GPT5_1
    ),
    ReasoningCapabilityRule(
        provider="openai",
        exact_model="gpt-5.2",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openai",
        model_prefix="gpt-5.2-",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openai",
        exact_model="gpt-5.3",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openai",
        model_prefix="gpt-5.3-",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openai",
        exact_model="gpt-5.4",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openai",
        model_prefix="gpt-5.4-",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5",
        capabilities=_OPENAI_CAPS_GPT5,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5-mini",
        capabilities=_OPENAI_CAPS_GPT5,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5.1",
        capabilities=_OPENAI_CAPS_GPT5_1,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5.2",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5.3",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
    ReasoningCapabilityRule(
        provider="openrouter",
        exact_model="openai/gpt-5.4",
        capabilities=_OPENAI_CAPS_GPT5_2_PLUS,
    ),
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
        provider="claude-code-minimax",
        model_prefix="MiniMax-",
        capabilities=_CLAUDE_HEADLESS_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="claude-code-kimi",
        model_prefix="kimi-",
        capabilities=_CLAUDE_HEADLESS_CAPS,
    ),
    ReasoningCapabilityRule(
        provider="codex", model_prefix="gpt-", capabilities=_CODEX_HEADLESS_CAPS
    ),
)


def reasoning_capabilities_for_model(
    *,
    provider: str,
    model: str,
) -> ReasoningCapabilities | None:
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
