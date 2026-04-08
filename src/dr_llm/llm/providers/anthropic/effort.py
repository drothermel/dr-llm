from __future__ import annotations

from dr_llm.llm.providers.effort_types import EffortSpec

ANTHROPIC_EFFORT_SUPPORTED_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
]

_ANTHROPIC_EFFORT_SUPPORTED_SET = frozenset(ANTHROPIC_EFFORT_SUPPORTED_MODELS)


def _supports_anthropic_effort(model: str) -> bool:
    return (
        model.startswith("claude-opus-4-6")
        or model.startswith("claude-sonnet-4-6")
        or model in _ANTHROPIC_EFFORT_SUPPORTED_SET
    )


def supported_effort_levels_for_anthropic(model: str) -> tuple[EffortSpec, ...]:
    if not _supports_anthropic_effort(model):
        return ()
    levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    if model.startswith("claude-opus-4-6"):
        levels.append(EffortSpec.MAX)
    return tuple(levels)
