from __future__ import annotations

from dr_llm.llm.names import EffortSpec, ReasoningMode
from dr_llm.llm.providers.impls.anthropic.effort import (
    ANTHROPIC_EFFORT_SUPPORTED_MODELS,
)
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT

_CLAUDE_HEADLESS_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.CLAUDE_CLI_EFFORT
)
_CLAUDE_HEADLESS_EFFORT_SUPPORTED_SET = frozenset(
    ANTHROPIC_EFFORT_SUPPORTED_MODELS
)

CLAUDE_HEADLESS_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        model_prefix="claude-", capabilities=_CLAUDE_HEADLESS_CAPS
    ),
)


def reasoning_capabilities_for_claude_code(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(CLAUDE_HEADLESS_CAPABILITY_RULES, model)


def supported_effort_levels_for_claude_code(
    model: str,
) -> tuple[EffortSpec, ...]:
    # Claude Code reasoning capabilities are prefix-based via
    # `CLAUDE_HEADLESS_CAPABILITY_RULES`, but effort support is intentionally
    # narrower and limited to `_CLAUDE_HEADLESS_EFFORT_SUPPORTED_SET`.
    if model not in _CLAUDE_HEADLESS_EFFORT_SUPPORTED_SET:
        return ()
    return FULL_EFFORT
