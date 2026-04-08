from __future__ import annotations

import pytest

from dr_llm.llm.providers.anthropic.effort import supported_effort_levels_for_anthropic
from dr_llm.llm.providers.effort_types import EffortSpec

_LOW_MEDIUM_HIGH = (EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH)
_LOW_MEDIUM_HIGH_MAX = (
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
)


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-6-20260101",
    ],
)
def test_supported_effort_levels_for_anthropic_opus_4_6_includes_max(model: str) -> None:
    assert supported_effort_levels_for_anthropic(model) == _LOW_MEDIUM_HIGH_MAX


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-sonnet-4-6-20260101",
        "claude-opus-4-5-20251101",
    ],
)
def test_supported_effort_levels_for_anthropic_non_opus_4_6_is_low_medium_high_only(
    model: str,
) -> None:
    assert supported_effort_levels_for_anthropic(model) == _LOW_MEDIUM_HIGH


def test_supported_effort_levels_for_anthropic_unknown_model_returns_empty() -> None:
    assert supported_effort_levels_for_anthropic("claude-haiku-3-5") == ()
