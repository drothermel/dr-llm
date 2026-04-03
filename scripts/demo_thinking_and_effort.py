#!/usr/bin/env python3
"""Demo: live verification of OpenAI/Codex/GLM thinking-level controls.

Usage:
  uv run python scripts/demo_thinking_and_effort.py
  uv run python scripts/demo_thinking_and_effort.py --provider openai
  uv run python scripts/demo_thinking_and_effort.py --provider codex
  uv run python scripts/demo_thinking_and_effort.py --provider glm
"""

from __future__ import annotations

from collections import defaultdict

import typer
from pydantic import BaseModel, ValidationError

from dr_llm.providers import build_default_registry
from dr_llm.providers.headless.codex_thinking import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
    codex_supports_xhigh_thinking,
)
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message
from dr_llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
    openai_supports_xhigh_thinking,
)
from dr_llm.providers.reasoning import (
    CodexReasoning,
    GlmReasoning,
    OpenAIReasoning,
    ThinkingLevel,
)
from dr_llm.providers.reasoning_capabilities import reasoning_capabilities_for_model
from dr_llm.providers.registry import ProviderRegistry

app = typer.Typer()

PROMPT = "Reply with exactly OK."
OPENAI_MODELS = [
    "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
]
CODEX_MODELS = [
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5.3-codex-spark",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "gpt-5.4-mini",
    "gpt-5.1-codex-mini",
]
GLM_MODELS = [
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5-turbo",
    "glm-5.1",
]
PROVIDER_MODELS = {
    "openai": OPENAI_MODELS,
    "codex": CODEX_MODELS,
    "glm": GLM_MODELS,
}
PHASES = ["models", "thinking"]


class SummaryCounts(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    had_output_text: int = 0


def supported_thinking_levels(provider: str, model: str) -> list[ThinkingLevel]:
    if provider == "openai":
        return _supported_openai_thinking_levels(model)
    if provider == "codex":
        return _supported_codex_thinking_levels(model)
    if provider == "glm":
        return _supported_glm_thinking_levels(model)
    raise ValueError(f"unsupported provider: {provider!r}")


def _supported_openai_thinking_levels(model: str) -> list[ThinkingLevel]:
    return _supported_openai_style_thinking_levels(
        supports_configurable=openai_supports_configurable_thinking(model),
        supports_off=openai_supports_off_thinking(model),
        supports_minimal=openai_supports_minimal_thinking(model),
        supports_xhigh=openai_supports_xhigh_thinking(model),
    )


def _supported_codex_thinking_levels(model: str) -> list[ThinkingLevel]:
    return _supported_openai_style_thinking_levels(
        supports_configurable=codex_supports_configurable_thinking(model),
        supports_off=codex_supports_off_thinking(model),
        supports_minimal=codex_supports_minimal_thinking(model),
        supports_xhigh=codex_supports_xhigh_thinking(model),
    )


def _supported_glm_thinking_levels(model: str) -> list[ThinkingLevel]:
    capabilities = reasoning_capabilities_for_model(provider="glm", model=model)
    if capabilities is None or capabilities.mode != "glm":
        return [ThinkingLevel.NA]
    return [ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE]


def _supported_openai_style_thinking_levels(
    *,
    supports_configurable: bool,
    supports_off: bool,
    supports_minimal: bool,
    supports_xhigh: bool,
) -> list[ThinkingLevel]:
    if not supports_configurable:
        return [ThinkingLevel.NA]
    levels = [ThinkingLevel.NA]
    if supports_off:
        levels.append(ThinkingLevel.OFF)
    elif supports_minimal:
        levels.append(ThinkingLevel.MINIMAL)
    levels.extend(
        [
            ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH,
        ]
    )
    if supports_xhigh:
        levels.append(ThinkingLevel.XHIGH)
    return levels


def default_thinking_for_model(provider: str, model: str) -> ThinkingLevel:
    levels = supported_thinking_levels(provider, model)
    if ThinkingLevel.OFF in levels:
        return ThinkingLevel.OFF
    if ThinkingLevel.MINIMAL in levels:
        return ThinkingLevel.MINIMAL
    if ThinkingLevel.LOW in levels:
        return ThinkingLevel.LOW
    return ThinkingLevel.NA


def reasoning_for_level(
    provider: str,
    thinking_level: ThinkingLevel,
) -> OpenAIReasoning | CodexReasoning | GlmReasoning | None:
    if thinking_level == ThinkingLevel.NA:
        return None
    if provider == "openai":
        return OpenAIReasoning(thinking_level=thinking_level)
    if provider == "codex":
        return CodexReasoning(thinking_level=thinking_level)
    if provider == "glm":
        return GlmReasoning(thinking_level=thinking_level)
    raise ValueError(f"unsupported provider: {provider!r}")


def format_attempt(
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> str:
    return f"{provider} | {model} | thinking={thinking_level.name}"


def availability_detail(missing: tuple[str, ...]) -> str:
    if not missing:
        return ""
    return ", ".join(missing)


def ensure_required_providers_available(providers: list[str]) -> None:
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
) -> LlmRequest:
    return LlmRequest(
        provider=provider,
        model=model,
        messages=[Message(role="user", content=PROMPT)],
        reasoning=reasoning_for_level(provider, thinking_level),
    )


def run_attempt(
    *,
    registry: ProviderRegistry,
    provider: str,
    model: str,
    phase: str,
    thinking_level: ThinkingLevel,
    counts: dict[tuple[str, str], SummaryCounts],
) -> None:
    key = (provider, phase)
    summary = counts[key]
    summary.attempted += 1
    print(format_attempt(provider, model, thinking_level))

    try:
        request = make_request(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
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
    providers: list[str],
) -> None:
    print("\n== models ==")
    for provider in providers:
        for model in PROVIDER_MODELS[provider]:
            run_attempt(
                registry=registry,
                provider=provider,
                model=model,
                phase="models",
                thinking_level=default_thinking_for_model(provider, model),
                counts=counts,
            )


def run_thinking_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[str],
) -> None:
    print("\n== thinking ==")
    for provider in providers:
        for model in PROVIDER_MODELS[provider]:
            for thinking_level in supported_thinking_levels(provider, model):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="thinking",
                    thinking_level=thinking_level,
                    counts=counts,
                )


def print_summary(
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[str],
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
        help="Providers to include in the sweep. Repeatable.",
    ),
) -> None:
    providers = provider or sorted(PROVIDER_MODELS)
    unsupported = [name for name in providers if name not in PROVIDER_MODELS]
    if unsupported:
        raise typer.BadParameter(
            f"Unsupported provider(s): {', '.join(sorted(unsupported))}"
        )

    ensure_required_providers_available(providers)
    counts: dict[tuple[str, str], SummaryCounts] = defaultdict(SummaryCounts)
    registry = build_default_registry()
    try:
        run_model_sweep(registry, counts, providers)
        run_thinking_sweep(registry, counts, providers)
        print_summary(counts, providers)
    finally:
        registry.close()


if __name__ == "__main__":
    app()
