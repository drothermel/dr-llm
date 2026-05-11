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
import typer
from pydantic import ValidationError

from dr_llm.demo import (
    ATTEMPT_SUMMARY_FIELDS,
    DEMO_THINKING_SWEEP_MODELS,
    DemoCounts,
    DemoPrompts,
    print_list,
)
from dr_llm.llm import (
    EffortSpec,
    LlmRequest,
    Message,
    OpenRouterReasoning,
    ProviderRegistry,
    ProviderName,
    ReasoningSpec,
    ThinkingLevel,
    build_default_registry,
)

app = typer.Typer()

SUPPORTED_PROVIDER_NAMES = ", ".join(
    sorted(provider.value for provider in DEMO_THINKING_SWEEP_MODELS)
)
PROMPT = DemoPrompts.EXACT_OK
PHASES = ["models", "thinking", "effort"]


def budget_tokens_for_level(
    registry: ProviderRegistry,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> int | None:
    if thinking_level != ThinkingLevel.BUDGET:
        return None
    return (
        registry.get(provider)
        .model_capabilities(model)
        .reasoning.min_budget_tokens
    )


def format_attempt(
    registry: ProviderRegistry,
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
    budget_tokens = budget_tokens_for_level(
        registry, provider, model, thinking_level
    )
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


def ensure_required_providers_available(
    registry: ProviderRegistry,
    providers: list[ProviderName],
) -> None:
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
        print_list(
            "Missing provider requirements:",
            missing,
            use_step=False,
        )
        raise typer.Exit(1)


def make_request(
    registry: ProviderRegistry,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
    *,
    explicit_reasoning: bool = False,
    reasoning_override: ReasoningSpec | None = None,
) -> LlmRequest:
    orchestrator = registry.get(provider)
    defaults = orchestrator.request_defaults(model)
    max_tokens = defaults.max_tokens
    reasoning = reasoning_override or registry.get(
        provider
    ).reasoning_for_thinking_level(
        model=model,
        thinking_level=thinking_level,
        budget_tokens=budget_tokens_for_level(
            registry, provider, model, thinking_level
        ),
    )
    if explicit_reasoning and reasoning is None:
        reasoning = defaults.reasoning
    return orchestrator.build_request(
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
    counts: dict[tuple[str, str], DemoCounts],
    reasoning_override: ReasoningSpec | None = None,
) -> None:
    key = (provider, phase)
    summary = counts[key]
    summary.increment("attempted")
    print(
        format_attempt(
            registry,
            provider,
            model,
            thinking_level,
            effort,
            reasoning_override,
        )
    )

    try:
        request = make_request(
            registry,
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            effort=effort,
            explicit_reasoning=explicit_reasoning,
            reasoning_override=reasoning_override,
        )
    except (ValidationError, ValueError) as exc:
        summary.increment("failed")
        print(f"  validation failure: {exc}")
        return

    orchestrator = registry.get(provider)
    try:
        response = orchestrator.generate(request)
    except Exception as exc:  # noqa: BLE001
        summary.increment("failed")
        print(f"  runtime failure: {type(exc).__name__}: {exc}")
        return

    summary.increment("succeeded")
    if response.text:
        summary.increment("had_output_text")
    print(f"  text: {response.text!r}")


def requires_explicit_reasoning(
    registry: ProviderRegistry, provider: str, model: str
) -> bool:
    return registry.get(provider).request_defaults(model).reasoning is not None


def run_model_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], DemoCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== models ==")
    for provider in providers:
        for model in DEMO_THINKING_SWEEP_MODELS[provider]:
            controls = registry.get(provider).reasoning_controls(model)
            run_attempt(
                registry=registry,
                provider=provider,
                model=model,
                phase="models",
                thinking_level=controls.default_thinking_level,
                effort=controls.default_effort,
                explicit_reasoning=requires_explicit_reasoning(
                    registry, provider, model
                ),
                counts=counts,
                reasoning_override=controls.default_reasoning,
            )


def run_thinking_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], DemoCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== thinking ==")
    for provider in providers:
        if provider == ProviderName.OPENROUTER:
            continue
        for model in DEMO_THINKING_SWEEP_MODELS[provider]:
            controls = registry.get(provider).reasoning_controls(model)
            for thinking_level in controls.supported_thinking_levels:
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="thinking",
                    thinking_level=thinking_level,
                    effort=controls.default_effort,
                    explicit_reasoning=True,
                    counts=counts,
                )


def run_effort_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], DemoCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== effort ==")
    for provider in providers:
        if provider == ProviderName.OPENROUTER:
            continue
        for model in DEMO_THINKING_SWEEP_MODELS[provider]:
            controls = registry.get(provider).reasoning_controls(model)
            for effort in controls.supported_effort_levels:
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="effort",
                    thinking_level=controls.default_thinking_level,
                    effort=effort,
                    explicit_reasoning=requires_explicit_reasoning(
                        registry, provider, model
                    ),
                    counts=counts,
                )


def print_summary(
    counts: dict[tuple[str, str], DemoCounts],
    providers: list[ProviderName],
) -> None:
    print("\n== summary ==")
    for provider in providers:
        print(provider)
        for phase in PHASES:
            summary = counts[(provider, phase)]
            print(f"  {phase}: {summary.format_line(ATTEMPT_SUMMARY_FIELDS)}")


def _resolve_sweep_providers(
    provider: list[str] | None,
) -> list[ProviderName]:
    supported_provider_values = {
        provider.value for provider in DEMO_THINKING_SWEEP_MODELS
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
    if provider:
        return [ProviderName(name) for name in provider]
    return sorted(DEMO_THINKING_SWEEP_MODELS)


def _run_thinking_and_effort_sweep(providers: list[ProviderName]) -> None:
    counts: dict[tuple[str, str], DemoCounts] = defaultdict(DemoCounts)
    registry = build_default_registry()
    try:
        ensure_required_providers_available(registry, providers)
        run_model_sweep(registry, counts, providers)
        run_thinking_sweep(registry, counts, providers)
        run_effort_sweep(registry, counts, providers)
        print_summary(counts, providers)
    finally:
        registry.close()


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
    providers = _resolve_sweep_providers(provider)
    _run_thinking_and_effort_sweep(providers)


if __name__ == "__main__":
    app()
