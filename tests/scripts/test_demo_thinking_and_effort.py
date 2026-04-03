from __future__ import annotations

import runpy
from pathlib import Path

from dr_llm.providers.reasoning import GlmReasoning, ThinkingLevel


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "demo_thinking_and_effort.py"
)
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH))


def test_glm_models_match_live_sync_list_snapshot() -> None:
    assert SCRIPT_GLOBALS["GLM_MODELS"] == [
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.6",
        "glm-4.7",
        "glm-5",
        "glm-5-turbo",
        "glm-5.1",
    ]


def test_glm_uses_explicit_thinking_levels_only() -> None:
    supported_thinking_levels = SCRIPT_GLOBALS["supported_thinking_levels"]
    default_thinking_for_model = SCRIPT_GLOBALS["default_thinking_for_model"]
    reasoning_for_level = SCRIPT_GLOBALS["reasoning_for_level"]

    levels = supported_thinking_levels("glm", "glm-4.5")
    assert levels == [ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE]
    assert default_thinking_for_model("glm", "glm-4.5") == ThinkingLevel.OFF

    off_reasoning = reasoning_for_level("glm", ThinkingLevel.OFF)
    adaptive_reasoning = reasoning_for_level("glm", ThinkingLevel.ADAPTIVE)
    assert isinstance(off_reasoning, GlmReasoning)
    assert off_reasoning.thinking_level == ThinkingLevel.OFF
    assert isinstance(adaptive_reasoning, GlmReasoning)
    assert adaptive_reasoning.thinking_level == ThinkingLevel.ADAPTIVE
