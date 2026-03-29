from __future__ import annotations

from dr_llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.providers.headless.reasoning import (
    ClaudeHeadlessReasoningConfig,
    CodexHeadlessReasoningConfig,
)
from dr_llm.providers.models import CallMode, ReasoningWarningCode
from dr_llm.providers.reasoning import ReasoningConfig


def test_codex_headless_reasoning_emits_unsupported_warning() -> None:
    result = CodexHeadlessReasoningConfig.from_base(
        ReasoningConfig(effort="high"),
        provider="codex",
        mode=CallMode.headless,
    )
    assert result.to_cli_args() == []
    assert len(result.warnings) == 1
    assert result.warnings[0].code == ReasoningWarningCode.unsupported_for_provider


def test_claude_headless_maps_max_tokens_to_effort_with_warning() -> None:
    result = ClaudeHeadlessReasoningConfig.from_base(
        ReasoningConfig(max_tokens=5000),
        provider="claude-code",
        mode=CallMode.headless,
    )
    assert result.to_cli_args() == ["--effort", "high"]
    assert result.warnings
    assert result.warnings[0].code == ReasoningWarningCode.mapped_with_heuristic


def test_google_reasoning_exclude_produces_warning() -> None:
    result = GoogleReasoningConfig.from_base(
        ReasoningConfig(effort="low", exclude=True),
        provider="google",
        mode=CallMode.api,
    )
    assert result.to_payload().get("thinkingLevel") == "low"
    assert result.warnings
    assert result.warnings[0].code == ReasoningWarningCode.partially_supported
