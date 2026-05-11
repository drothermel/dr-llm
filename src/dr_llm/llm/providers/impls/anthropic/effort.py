from __future__ import annotations

from dr_llm.llm.names import EffortSpec
from dr_llm.llm.providers.concepts.thinking_utils import matches_family
from dr_llm.llm.providers.impls.anthropic.families import (
    ANTHROPIC_EFFORT_SUPPORTED_MODELS,
    AnthropicModelFamily,
)


def _supports_anthropic_effort(model: str) -> bool:
    return matches_family(
        normalized=model, families=ANTHROPIC_EFFORT_SUPPORTED_MODELS
    )


def supported_effort_levels_for_anthropic(
    model: str,
) -> tuple[EffortSpec, ...]:
    if not _supports_anthropic_effort(model):
        return ()
    levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    if model.startswith(AnthropicModelFamily.CLAUDE_OPUS_46):
        levels.append(EffortSpec.MAX)
    return tuple(levels)
