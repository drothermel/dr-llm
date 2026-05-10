#!/usr/bin/env python3
"""Demo: live verification of provider thinking and effort controls.

Prerequisites:
  API keys or CLI tools for every provider under test.

Usage:
  uv run python scripts/demo_thinking_and_effort.py
  uv run python scripts/demo_thinking_and_effort.py --provider openrouter
  uv run python scripts/demo_thinking_and_effort.py --provider openai
  uv run python scripts/demo_thinking_and_effort.py --provider codex
  uv run python scripts/demo_thinking_and_effort.py --provider google
  uv run python scripts/demo_thinking_and_effort.py --provider claude-code
  uv run python scripts/demo_thinking_and_effort.py --provider minimax
  uv run python scripts/demo_thinking_and_effort.py --provider kimi-code
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import typer
from pydantic import BaseModel, ValidationError

from dr_llm.demo import DEMO_PROVIDER_MODELS
from dr_llm.llm import (
    ApiLlmRequest,
    SamplingApiProviderName,
    EffortSpec,
    HeadlessLlmRequest,
    HeadlessProviderName,
    KimiCodeLlmRequest,
    KimiCodeProviderName,
    LlmRequest,
    Message,
    OpenRouterReasoning,
    ProviderCategories,
    ProviderRegistry,
    ProviderName,
    ReasoningSpec,
    ThinkingLevel,
    build_default_registry,
    default_effort,
    default_reasoning,
    default_thinking_level,
    reasoning_capabilities_for_model,
    reasoning_for_thinking_level,
    supported_effort_levels,
    supported_thinking_levels,
)

app = typer.Typer()

SUPPORTED_PROVIDER_NAMES = ", ".join(
    sorted(provider.value for provider in DEMO_PROVIDER_MODELS)
)
PROMPT = "Reply with exactly OK."
KIMI_CODE_MAX_TOKENS = 2048
PHASES = ["models", "thinking", "effort"]


class SummaryCounts(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    had_output_text: int = 0


def budget_tokens_for_level(
    provider: str, model: str, thinking_level: ThinkingLevel
) -> int | None:
    if thinking_level != ThinkingLevel.BUDGET:
        return None
    capabilities = reasoning_capabilities_for_model(
        provider=provider, model=model
    )
    if capabilities is None or capabilities.min_budget_tokens is None:
        return None
    return capabilities.min_budget_tokens


def format_attempt(
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
    reasoning_override: ReasoningSpec | None = None,
) -> str:
    if provider == ProviderName.OPENROUTER:
        detail = f"{provider} | {model}"
        match reasoning_override:
            case OpenRouterReasoning(effort=override_effort) if (
                override_effort is not None
            ):
                detail += f" | reasoning=effort({override_effort})"
            case OpenRouterReasoning(enabled=enabled) if enabled is not None:
                detail += f" | reasoning=enabled({str(enabled).lower()})"
            case _:
                detail += " | reasoning=none"
        return detail
    detail = f"{provider} | {model} | thinking={thinking_level.name}"
    budget_tokens = budget_tokens_for_level(provider, model, thinking_level)
    if budget_tokens is not None:
        detail = (
            f"{provider} | {model} | "
            f"thinking={thinking_level.name}({budget_tokens})"
        )
    if effort != EffortSpec.NA:
        detail += f" | effort={effort.name}"
    return detail


def availability_detail(missing: tuple[str, ...]) -> str:
    if not missing:
        return ""
    return ", ".join(missing)


def ensure_required_providers_available(providers: list[ProviderName]) -> None:
    registry = build_default_registry()
    try:
        missing: list[str] = []
        for provider in providers:
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
    *,
    explicit_reasoning: bool = False,
    reasoning_override: ReasoningSpec | None = None,
) -> LlmRequest:
    max_tokens = (
        KIMI_CODE_MAX_TOKENS if provider == ProviderName.KIMI_CODE else None
    )
    reasoning = reasoning_override or reasoning_for_thinking_level(
        provider=provider,
        model=model,
        thinking_level=thinking_level,
        budget_tokens=budget_tokens_for_level(provider, model, thinking_level),
        explicit_na=explicit_reasoning,
    )
    if provider in ProviderCategories().headless:
        return HeadlessLlmRequest(
            provider=cast(HeadlessProviderName, provider),
            model=model,
            messages=[Message(role="user", content=PROMPT)],
            effort=effort,
            reasoning=reasoning,
        )
    if provider == ProviderName.KIMI_CODE:
        return KimiCodeLlmRequest(
            provider=cast(KimiCodeProviderName, provider),
            model=model,
            messages=[Message(role="user", content=PROMPT)],
            max_tokens=KIMI_CODE_MAX_TOKENS,
            effort=effort,
            reasoning=reasoning,
        )
    return ApiLlmRequest(
        provider=cast(SamplingApiProviderName, provider),
        model=model,
        messages=[Message(role="user", content=PROMPT)],
        max_tokens=max_tokens,
        effort=effort,
        reasoning=reasoning,
    )


def run_attempt(
    *,
    registry: ProviderRegistry,
    provider: str,
    model: str,
    phase: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
    explicit_reasoning: bool,
    counts: dict[tuple[str, str], SummaryCounts],
    reasoning_override: ReasoningSpec | None = None,
) -> None:
    key = (provider, phase)
    summary = counts[key]
    summary.attempted += 1
    print(
        format_attempt(
            provider,
            model,
            thinking_level,
            effort,
            reasoning_override,
        )
    )

    try:
        request = make_request(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            effort=effort,
            explicit_reasoning=explicit_reasoning,
            reasoning_override=reasoning_override,
        )
    except (ValidationError, ValueError) as exc:
        summary.failed += 1
        print(f"  validation failure: {exc}")
        return

    model_provider = registry.get(provider)
    try:
        response = model_provider.generate(request)
    except Exception as exc:  # noqa: BLE001
        summary.failed += 1
        print(f"  runtime failure: {type(exc).__name__}: {exc}")
        return

    summary.succeeded += 1
    if response.text:
        summary.had_output_text += 1
    print(f"  text: {response.text!r}")


def requires_explicit_reasoning(provider: str) -> bool:
    return provider == ProviderName.MINIMAX


def run_model_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== models ==")
    for provider in providers:
        for model in DEMO_PROVIDER_MODELS[provider]:
            run_attempt(
                registry=registry,
                provider=provider,
                model=model,
                phase="models",
                thinking_level=default_thinking_level(
                    provider=provider, model=model
                ),
                effort=default_effort(provider=provider, model=model),
                explicit_reasoning=requires_explicit_reasoning(provider),
                counts=counts,
                reasoning_override=default_reasoning(
                    provider=provider, model=model
                ),
            )


def run_thinking_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== thinking ==")
    for provider in providers:
        if provider == ProviderName.OPENROUTER:
            continue
        for model in DEMO_PROVIDER_MODELS[provider]:
            for thinking_level in supported_thinking_levels(
                provider=provider, model=model
            ):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="thinking",
                    thinking_level=thinking_level,
                    effort=default_effort(provider=provider, model=model),
                    explicit_reasoning=True,
                    counts=counts,
                )


def run_effort_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== effort ==")
    for provider in providers:
        if provider == ProviderName.OPENROUTER:
            continue
        for model in DEMO_PROVIDER_MODELS[provider]:
            for effort in supported_effort_levels(
                provider=provider, model=model
            ):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="effort",
                    thinking_level=default_thinking_level(
                        provider=provider, model=model
                    ),
                    effort=effort,
                    explicit_reasoning=requires_explicit_reasoning(provider),
                    counts=counts,
                )


def print_summary(
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== summary ==")
    for provider in providers:
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
def main(
    provider: list[str] | None = typer.Option(
        None,
        "--provider",
        help=(
            "Providers to include in the sweep. Repeatable. "
            f"Supported: {SUPPORTED_PROVIDER_NAMES}."
        ),
    ),
) -> None:
    """Sweep curated models for provider-specific reasoning and effort support."""
    supported_provider_values = {
        provider.value for provider in DEMO_PROVIDER_MODELS
    }
    unsupported = [
        name
        for name in provider or []
        if name not in supported_provider_values
    ]
    if unsupported:
        raise typer.BadParameter(
            f"Unsupported provider(s): {', '.join(sorted(unsupported))}"
        )
    providers = (
        [ProviderName(name) for name in provider]
        if provider
        else sorted(DEMO_PROVIDER_MODELS)
    )

    ensure_required_providers_available(providers)
    counts: dict[tuple[str, str], SummaryCounts] = defaultdict(SummaryCounts)
    registry = build_default_registry()
    try:
        run_model_sweep(registry, counts, providers)
        run_thinking_sweep(registry, counts, providers)
        run_effort_sweep(registry, counts, providers)
        print_summary(counts, providers)
    finally:
        registry.close()


if __name__ == "__main__":
    app()
