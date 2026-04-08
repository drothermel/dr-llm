from __future__ import annotations

from dr_llm.llm.providers.anthropic.effort import ANTHROPIC_EFFORT_SUPPORTED_MODELS
from dr_llm.llm.providers.effort_types import FULL_EFFORT, EffortSpec
from dr_llm.llm.providers.reasoning_capability_types import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)

_CLAUDE_HEADLESS_CAPS = ReasoningCapabilities(mode="claude_cli_effort")
_CLAUDE_HEADLESS_EFFORT_SUPPORTED_SET = frozenset(ANTHROPIC_EFFORT_SUPPORTED_MODELS)

CLAUDE_HEADLESS_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(model_prefix="claude-", capabilities=_CLAUDE_HEADLESS_CAPS),
)


def reasoning_capabilities_for_claude_code(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(CLAUDE_HEADLESS_CAPABILITY_RULES, model)


def supported_effort_levels_for_claude_code(model: str) -> tuple[EffortSpec, ...]:
    if model not in _CLAUDE_HEADLESS_EFFORT_SUPPORTED_SET:
        return ()
    return FULL_EFFORT
