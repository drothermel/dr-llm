from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.model_family import ModelFamily


class GoogleThinkingLevel(StrEnum):
    MINIMAL = ThinkingLevel.MINIMAL
    LOW = ThinkingLevel.LOW
    MEDIUM = ThinkingLevel.MEDIUM
    HIGH = ThinkingLevel.HIGH


class ReasoningCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ReasoningMode = ReasoningMode.UNSUPPORTED
    google_levels: tuple[GoogleThinkingLevel, ...] = ()
    min_budget_tokens: int | None = None
    max_budget_tokens: int | None = None
    supports_dynamic: bool = False

    @property
    def supports_reasoning(self) -> bool:
        return self.mode != ReasoningMode.UNSUPPORTED


class ReasoningCapabilityRule(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    family: ModelFamily
    capabilities: ReasoningCapabilities


def resolve_capability_rules(
    rules: tuple[ReasoningCapabilityRule, ...], model: str
) -> ReasoningCapabilities | None:
    sorted_rules = sorted(
        rules,
        key=lambda rule: len(str(rule.family)),
        reverse=True,
    )
    for rule in sorted_rules:
        if rule.family.in_family(model):
            return rule.capabilities
    return None


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: ReasoningCapabilities = ReasoningCapabilities(
        mode=ReasoningMode.UNSUPPORTED
    )
    supported_effort_levels: tuple[EffortSpec, ...] = ()


def build_model_capabilities(
    *,
    reasoning: ReasoningCapabilities | None,
    supported_effort_levels: tuple[EffortSpec, ...] = (),
) -> ModelCapabilities:
    resolved_reasoning = reasoning or ReasoningCapabilities(
        mode=ReasoningMode.UNSUPPORTED
    )
    return ModelCapabilities(
        reasoning=resolved_reasoning,
        supported_effort_levels=supported_effort_levels,
    )
