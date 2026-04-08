from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

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


def resolve_capability_rules(
    rules: tuple[ReasoningCapabilityRule, ...], model: str
) -> ReasoningCapabilities | None:
    for rule in rules:
        if rule.exact_model is not None and model == rule.exact_model:
            return rule.capabilities
    prefix_rules = sorted(
        (rule for rule in rules if rule.model_prefix is not None),
        key=lambda rule: len(rule.model_prefix or ""),
        reverse=True,
    )
    for rule in prefix_rules:
        prefix = rule.model_prefix or ""
        if model.startswith(prefix):
            return rule.capabilities
    return None
