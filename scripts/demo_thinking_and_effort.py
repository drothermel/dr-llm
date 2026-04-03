#!/usr/bin/env python3
"""Demo: live verification of Anthropic/Claude Code thinking and effort shapes.

Usage:
  uv run python scripts/demo_thinking_and_effort.py
"""

from __future__ import annotations

from collections import defaultdict

import typer
from pydantic import BaseModel, ValidationError

from dr_llm.providers import build_default_registry
from dr_llm.providers.anthropic.effort import ANTHROPIC_EFFORT_SUPPORTED_MODELS
from dr_llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
    ANTHROPIC_BUDGET_THINKING_SUPPORTED,
)
from dr_llm.providers.effort import EffortSpec
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.providers.reasoning import AnthropicReasoning, ThinkingLevel

app = typer.Typer()

PROMPT = "Reply with exactly OK."
BUDGET_TOKENS = 2048
THINKING_LEVELS = [
    ThinkingLevel.NA,
    ThinkingLevel.OFF,
    ThinkingLevel.BUDGET,
    ThinkingLevel.ADAPTIVE,
]
EFFORT_LEVELS = [
    EffortSpec.NA,
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]
CLAUDE_CODE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]
PROVIDER_MODELS = {
    "anthropic": ANTHROPIC_MODELS,
    "claude-code": CLAUDE_CODE_MODELS,
}
PHASES = ["models", "thinking", "effort"]


class SummaryCounts(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    had_output_text: int = 0


def anthropic_supported_thinking_levels(model: str) -> list[ThinkingLevel]:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return [ThinkingLevel.NA, ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE]
    if model in ANTHROPIC_BUDGET_THINKING_SUPPORTED:
        return [ThinkingLevel.NA, ThinkingLevel.OFF, ThinkingLevel.BUDGET]
    return [ThinkingLevel.NA, ThinkingLevel.OFF]


def anthropic_supported_effort_levels(model: str) -> list[EffortSpec]:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    return [EffortSpec.NA]


def anthropic_default_thinking_for_model(model: str) -> ThinkingLevel:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return ThinkingLevel.ADAPTIVE
    return ThinkingLevel.OFF


def anthropic_default_thinking_for_effort_sweep(model: str) -> ThinkingLevel:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return ThinkingLevel.ADAPTIVE
    return ThinkingLevel.OFF


def anthropic_default_effort_for_model(model: str) -> EffortSpec:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return EffortSpec.MEDIUM
    return EffortSpec.NA


def anthropic_default_effort_for_thinking_sweep(model: str) -> EffortSpec:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return EffortSpec.MEDIUM
    return EffortSpec.NA


def claude_code_supported_thinking_levels(model: str) -> list[ThinkingLevel]:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return [ThinkingLevel.ADAPTIVE]
    return [ThinkingLevel.NA]


def claude_code_supported_effort_levels(model: str) -> list[EffortSpec]:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    return [EffortSpec.NA]


def claude_code_default_thinking_for_model(model: str) -> ThinkingLevel:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return ThinkingLevel.ADAPTIVE
    return ThinkingLevel.NA


def claude_code_default_thinking_for_effort_sweep(model: str) -> ThinkingLevel:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return ThinkingLevel.ADAPTIVE
    return ThinkingLevel.NA


def claude_code_default_effort_for_model(model: str) -> EffortSpec:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return EffortSpec.MEDIUM
    return EffortSpec.NA


def claude_code_default_effort_for_thinking_sweep(model: str) -> EffortSpec:
    if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
        return EffortSpec.MEDIUM
    return EffortSpec.NA


def supported_thinking_levels(provider: str, model: str) -> list[ThinkingLevel]:
    if provider == "anthropic":
        return anthropic_supported_thinking_levels(model)
    if provider == "claude-code":
        return claude_code_supported_thinking_levels(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def supported_effort_levels(provider: str, model: str) -> list[EffortSpec]:
    if provider == "anthropic":
        return anthropic_supported_effort_levels(model)
    if provider == "claude-code":
        return claude_code_supported_effort_levels(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def reasoning_for_thinking_level(level: ThinkingLevel) -> AnthropicReasoning | None:
    if level == ThinkingLevel.NA:
        return None
    if level == ThinkingLevel.OFF:
        return AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    if level == ThinkingLevel.BUDGET:
        return AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=BUDGET_TOKENS,
        )
    if level == ThinkingLevel.ADAPTIVE:
        return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    raise ValueError(f"unsupported thinking level: {level!r}")


def default_thinking_for_model(provider: str, model: str) -> ThinkingLevel:
    if provider == "anthropic":
        return anthropic_default_thinking_for_model(model)
    if provider == "claude-code":
        return claude_code_default_thinking_for_model(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def default_thinking_for_effort_sweep(provider: str, model: str) -> ThinkingLevel:
    if provider == "anthropic":
        return anthropic_default_thinking_for_effort_sweep(model)
    if provider == "claude-code":
        return claude_code_default_thinking_for_effort_sweep(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def default_effort_for_model(provider: str, model: str) -> EffortSpec:
    if provider == "anthropic":
        return anthropic_default_effort_for_model(model)
    if provider == "claude-code":
        return claude_code_default_effort_for_model(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def default_effort_for_thinking_sweep(provider: str, model: str) -> EffortSpec:
    if provider == "anthropic":
        return anthropic_default_effort_for_thinking_sweep(model)
    if provider == "claude-code":
        return claude_code_default_effort_for_thinking_sweep(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def format_thinking(level: ThinkingLevel) -> str:
    if level == ThinkingLevel.BUDGET:
        return f"{level.name}({BUDGET_TOKENS})"
    return level.name


def format_effort(level: EffortSpec) -> str:
    return level.name


def format_attempt(
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
) -> str:
    return (
        f"{provider} | {model} | "
        f"thinking={format_thinking(thinking_level)} | "
        f"effort={format_effort(effort)}"
    )


def availability_detail(missing: tuple[str, ...]) -> str:
    if not missing:
        return ""
    return ", ".join(missing)


def ensure_required_providers_available() -> None:
    registry = build_default_registry()
    try:
        missing: list[str] = []
        for provider in ("anthropic", "claude-code"):
            status = registry.availability_status(provider)
            if status.available:
                continue
            reasons: list[str] = []
            if status.missing_env_vars:
                reasons.append(
                    f"missing env: {availability_detail(status.missing_env_vars)}"
                )
            if status.missing_executables:
                reasons.append(
                    "missing executable: "
                    f"{availability_detail(status.missing_executables)}"
                )
            missing.append(f"{provider}: {'; '.join(reasons)}")
        if missing:
            print("Missing provider requirements:")
            for detail in missing:
                print(f"  - {detail}")
            raise typer.Exit(1)
    finally:
        registry.close()


def make_request(
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
) -> LlmRequest:
    return LlmRequest(
        provider=provider,
        model=model,
        messages=[Message(role="user", content=PROMPT)],
        effort=effort,
        reasoning=reasoning_for_thinking_level(thinking_level),
    )


def run_attempt(
    *,
    registry: ProviderRegistry,
    provider: str,
    model: str,
    phase: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
    counts: dict[tuple[str, str], SummaryCounts],
) -> None:
    key = (provider, phase)
    summary = counts[key]
    summary.attempted += 1
    print(format_attempt(provider, model, thinking_level, effort))

    try:
        request = make_request(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            effort=effort,
        )
    except (ValidationError, ValueError) as exc:
        summary.failed += 1
        print(f"  validation failure: {exc}")
        return

    adapter = registry.get(provider)
    try:
        response = adapter.generate(request)
    except Exception as exc:  # noqa: BLE001
        summary.failed += 1
        print(f"  runtime failure: {type(exc).__name__}: {exc}")
        return

    summary.succeeded += 1
    if response.text:
        summary.had_output_text += 1
    print(f"  text: {response.text!r}")


def run_model_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
) -> None:
    print("\n== models ==")
    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            run_attempt(
                registry=registry,
                provider=provider,
                model=model,
                phase="models",
                thinking_level=default_thinking_for_model(provider, model),
                effort=default_effort_for_model(provider, model),
                counts=counts,
            )


def run_thinking_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
) -> None:
    print("\n== thinking ==")
    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            effort = default_effort_for_thinking_sweep(provider, model)
            for thinking_level in supported_thinking_levels(provider, model):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="thinking",
                    thinking_level=thinking_level,
                    effort=effort,
                    counts=counts,
                )


def run_effort_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
) -> None:
    print("\n== effort ==")
    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            thinking_level = default_thinking_for_effort_sweep(provider, model)
            for effort in supported_effort_levels(provider, model):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="effort",
                    thinking_level=thinking_level,
                    effort=effort,
                    counts=counts,
                )


def print_summary(counts: dict[tuple[str, str], SummaryCounts]) -> None:
    print("\n== summary ==")
    for provider in ("anthropic", "claude-code"):
        print(provider)
        for phase in PHASES:
            summary = counts[(provider, phase)]
            print(
                "  "
                f"{phase}: attempted={summary.attempted} "
                f"succeeded={summary.succeeded} "
                f"failed={summary.failed} "
                f"had_output_text={summary.had_output_text}"
            )


@app.command()
def main() -> None:
    ensure_required_providers_available()
    counts: dict[tuple[str, str], SummaryCounts] = defaultdict(SummaryCounts)
    registry = build_default_registry()
    try:
        run_model_sweep(registry, counts)
        run_thinking_sweep(registry, counts)
        run_effort_sweep(registry, counts)
        print_summary(counts)
    finally:
        registry.close()


if __name__ == "__main__":
    app()
