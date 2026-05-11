from __future__ import annotations

from dr_llm.llm.names import EffortSpec
from dr_llm.llm.providers.impls.anthropic.capabilities import (
    anthropic_supports_effort,
)
from dr_llm.llm.providers.impls.anthropic.families import (
    AnthropicModelFamily,
)


def supported_effort_levels_for_anthropic(
    model: str,
) -> tuple[EffortSpec, ...]:
    if not anthropic_supports_effort(model):
        return ()
    levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    if AnthropicModelFamily.CLAUDE_OPUS_46.in_family(model):
        levels.append(EffortSpec.MAX)
    return tuple(levels)
