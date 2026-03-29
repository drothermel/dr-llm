from __future__ import annotations

from dr_llm.providers.anthropic.reasoning import AnthropicReasoningConfig
from dr_llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.providers.headless.reasoning import ClaudeHeadlessReasoningConfig, CodexHeadlessReasoningConfig
from dr_llm.providers.models import CallMode, ReasoningWarningCode
from dr_llm.providers.openai_compat.reasoning import OpenAICompatReasoningConfig
from dr_llm.providers.reasoning import ReasoningConfig


# --- OpenAI-compat ---

def test_openai_compat_passes_through_config() -> None:
    config = ReasoningConfig(effort="high", exclude=False)
    result = OpenAICompatReasoningConfig.from_base(config, provider="openai", mode=CallMode.api)
    payload = result.to_payload()
    assert payload["effort"] == "high"
    assert payload["exclude"] is False
    assert result.warnings == []


def test_openai_compat_none_config_returns_empty() -> None:
    result = OpenAICompatReasoningConfig.from_base(None, provider="openai", mode=CallMode.api)
    assert result.to_payload() == {}
    assert result.warnings == []


# --- Anthropic ---

def test_anthropic_effort_computes_budget() -> None:
    config = ReasoningConfig(effort="high")
    result = AnthropicReasoningConfig.from_base(
        config, provider="anthropic", mode=CallMode.api, request_max_tokens=4096
    )
    payload = result.to_payload()
    assert payload.get("type") == "enabled"
    budget = payload["budget_tokens"]
    assert 1024 <= budget <= 128000
    assert result.warnings == []


def test_anthropic_clamps_budget_below_max_tokens() -> None:
    config = ReasoningConfig(max_tokens=100000)
    result = AnthropicReasoningConfig.from_base(
        config, provider="anthropic", mode=CallMode.api, request_max_tokens=2048
    )
    payload = result.to_payload()
    # budget should be clamped below request_max_tokens
    assert payload.get("budget_tokens", 0) < 2048
    warning_codes = [w.code for w in result.warnings]
    assert ReasoningWarningCode.mapped_with_heuristic in warning_codes


def test_anthropic_none_config_returns_empty() -> None:
    result = AnthropicReasoningConfig.from_base(
        None, provider="anthropic", mode=CallMode.api, request_max_tokens=4096
    )
    assert result.to_payload() == {}
    assert result.warnings == []


# --- Google ---

def test_google_maps_effort_to_thinking_level() -> None:
    config = ReasoningConfig(effort="low")
    result = GoogleReasoningConfig.from_base(config, provider="google", mode=CallMode.api)
    assert result.to_payload()["thinkingLevel"] == "low"
    assert result.warnings == []


def test_google_exclude_produces_warning() -> None:
    config = ReasoningConfig(exclude=True)
    result = GoogleReasoningConfig.from_base(config, provider="google", mode=CallMode.api)
    warning_codes = [w.code for w in result.warnings]
    assert ReasoningWarningCode.partially_supported in warning_codes


def test_google_disabled_sets_minimal() -> None:
    config = ReasoningConfig(enabled=False)
    result = GoogleReasoningConfig.from_base(config, provider="google", mode=CallMode.api)
    payload = result.to_payload()
    assert payload["thinkingBudget"] == 0
    assert payload["thinkingLevel"] == "minimal"


# --- Claude headless ---

def test_claude_headless_maps_max_tokens_to_effort() -> None:
    config = ReasoningConfig(max_tokens=5000)
    result = ClaudeHeadlessReasoningConfig.from_base(
        config, provider="claude", mode=CallMode.headless
    )
    assert result.to_cli_args() == ["--effort", "high"]
    warning_codes = [w.code for w in result.warnings]
    assert ReasoningWarningCode.mapped_with_heuristic in warning_codes


def test_claude_headless_maps_xhigh_to_high() -> None:
    config = ReasoningConfig(effort="xhigh")
    result = ClaudeHeadlessReasoningConfig.from_base(
        config, provider="claude", mode=CallMode.headless
    )
    assert result.to_cli_args() == ["--effort", "high"]


# --- Codex headless ---

def test_codex_headless_emits_unsupported_warning() -> None:
    config = ReasoningConfig(effort="high")
    result = CodexHeadlessReasoningConfig.from_base(
        config, provider="codex", mode=CallMode.headless
    )
    assert result.to_cli_args() == []
    warning_codes = [w.code for w in result.warnings]
    assert ReasoningWarningCode.unsupported_for_provider in warning_codes
