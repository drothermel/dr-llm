from __future__ import annotations

from dr_llm.reasoning import (
    map_reasoning_for_claude_headless,
    map_reasoning_for_codex_headless,
    map_reasoning_for_google,
)
from dr_llm.types import CallMode, ReasoningConfig, ReasoningWarningCode


def test_codex_headless_reasoning_emits_unsupported_warning() -> None:
    result = map_reasoning_for_codex_headless(
        ReasoningConfig(effort="high"),
        provider="codex",
        mode=CallMode.headless,
    )
    assert result.cli_args == []
    assert len(result.warnings) == 1
    assert result.warnings[0].code == ReasoningWarningCode.unsupported_for_provider


def test_claude_headless_maps_max_tokens_to_effort_with_warning() -> None:
    result = map_reasoning_for_claude_headless(
        ReasoningConfig(max_tokens=5000),
        provider="claude-code",
        mode=CallMode.headless,
    )
    assert result.cli_args == ["--effort", "high"]
    assert result.warnings
    assert result.warnings[0].code == ReasoningWarningCode.mapped_with_heuristic


def test_google_reasoning_exclude_produces_warning() -> None:
    result = map_reasoning_for_google(
        ReasoningConfig(effort="low", exclude=True),
        provider="google",
        mode=CallMode.api,
    )
    assert result.payload.get("thinkingLevel") == "low"
    assert result.warnings
    assert result.warnings[0].code == ReasoningWarningCode.partially_supported
