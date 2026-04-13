#!/usr/bin/env python3
"""Demo: live verification of provider thinking and effort controls.

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

from _demo_thinking_models import PROVIDER_MODELS
from dr_llm.llm.request import (
    ApiLlmRequest,
    ApiProviderName,
    HeadlessLlmRequest,
    HeadlessProviderName,
    KimiCodeLlmRequest,
    KimiCodeProviderName,
    LlmRequest,
)
from dr_llm.llm.providers.anthropic.thinking import (
    ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED,
)
from dr_llm.llm.providers.effort import EffortSpec, supported_effort_levels
from dr_llm.llm.providers.headless.codex_thinking import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
)
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningSpec,
    ThinkingLevel,
)
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model
from dr_llm.llm.providers.registry import ProviderRegistry, build_default_registry

app = typer.Typer()

PROMPT = "Reply with exactly OK."
GOOGLE_FIXED_BUDGET = 1024
KIMI_CODE_FIXED_BUDGET = 1024
KIMI_CODE_MAX_TOKENS = 2048
PHASES = ["models", "thinking", "effort"]


class SummaryCounts(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    had_output_text: int = 0


def supported_thinking_levels(provider: str, model: str) -> list[ThinkingLevel]:
    if provider == "claude-code":
        return _supported_claude_code_thinking_levels(model)
    if provider == "minimax":
        return [ThinkingLevel.NA]
    if provider == "kimi-code":
        return [ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE, ThinkingLevel.BUDGET]
    if provider == "openai":
        return _supported_openai_thinking_levels(model)
    if provider == "codex":
        return _supported_codex_thinking_levels(model)
    if provider == "google":
        return _supported_google_thinking_levels(model)
    if provider == "openrouter":
        return [ThinkingLevel.NA]
    raise ValueError(f"unsupported provider: {provider!r}")


def _supported_openai_thinking_levels(model: str) -> list[ThinkingLevel]:
    return _supported_openai_style_thinking_levels(
        supports_configurable=openai_supports_configurable_thinking(model),
        supports_off=openai_supports_off_thinking(model),
        supports_minimal=openai_supports_minimal_thinking(model),
    )


def _supported_codex_thinking_levels(model: str) -> list[ThinkingLevel]:
    levels = _supported_openai_style_thinking_levels(
        supports_configurable=codex_supports_configurable_thinking(model),
        supports_off=codex_supports_off_thinking(model),
        supports_minimal=codex_supports_minimal_thinking(model),
    )
    if codex_supports_configurable_thinking(model):
        levels.append(ThinkingLevel.XHIGH)
    return levels


def _supported_claude_code_thinking_levels(model: str) -> list[ThinkingLevel]:
    if model in ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED:
        return [ThinkingLevel.ADAPTIVE]
    return [ThinkingLevel.NA]


def _supported_google_thinking_levels(model: str) -> list[ThinkingLevel]:
    capabilities = reasoning_capabilities_for_model(provider="google", model=model)
    if capabilities is None or capabilities.mode == "unsupported":
        return [ThinkingLevel.NA]
    if capabilities.mode == "google_budget":
        return [ThinkingLevel.ADAPTIVE, ThinkingLevel.OFF, ThinkingLevel.BUDGET]
    if capabilities.mode == "google_level":
        return [ThinkingLevel(level) for level in capabilities.google_levels]
    raise ValueError(f"unexpected google reasoning mode: {capabilities.mode!r}")


def _supported_openai_style_thinking_levels(
    *,
    supports_configurable: bool,
    supports_off: bool,
    supports_minimal: bool,
) -> list[ThinkingLevel]:
    if not supports_configurable:
        return [ThinkingLevel.NA]
    levels: list[ThinkingLevel] = []
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
    return levels


def default_thinking_for_model(provider: str, model: str) -> ThinkingLevel:
    if provider in {"minimax", "kimi-code"}:
        return ThinkingLevel.NA
    if provider == "claude-code":
        levels = supported_thinking_levels(provider, model)
        if ThinkingLevel.ADAPTIVE in levels:
            return ThinkingLevel.ADAPTIVE
        return ThinkingLevel.NA
    levels = supported_thinking_levels(provider, model)
    if ThinkingLevel.OFF in levels:
        return ThinkingLevel.OFF
    if ThinkingLevel.MINIMAL in levels:
        return ThinkingLevel.MINIMAL
    if ThinkingLevel.LOW in levels:
        return ThinkingLevel.LOW
    return ThinkingLevel.NA


def default_effort_for_model(provider: str, model: str) -> EffortSpec:
    levels = supported_effort_levels(provider=provider, model=model)
    if levels:
        return levels[0]
    return EffortSpec.NA


def default_reasoning_override(
    provider: str,
    model: str,
) -> ReasoningSpec | None:
    if provider != "openrouter":
        return None
    policy = openrouter_model_policy(model)
    if policy is None:
        raise ValueError(f"missing openrouter policy for model={model!r}")
    if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
        return None
    if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        if policy.supports_disable:
            return OpenRouterReasoning(enabled=False)
        return OpenRouterReasoning(enabled=True)
    return OpenRouterReasoning(effort=policy.allowed_efforts[0])


def reasoning_for_level(
    provider: str,
    thinking_level: ThinkingLevel,
    *,
    explicit: bool = False,
) -> AnthropicReasoning | OpenAIReasoning | CodexReasoning | GoogleReasoning | None:
    if provider == "claude-code":
        if thinking_level == ThinkingLevel.ADAPTIVE:
            return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        if thinking_level == ThinkingLevel.NA and explicit:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(f"unsupported claude-code thinking level: {thinking_level!r}")
    if provider == "minimax":
        if thinking_level == ThinkingLevel.NA and explicit:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(f"unsupported minimax thinking level: {thinking_level!r}")
    if provider == "kimi-code":
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=KIMI_CODE_FIXED_BUDGET,
            )
        return AnthropicReasoning(thinking_level=thinking_level)
    if thinking_level == ThinkingLevel.NA:
        return None
    if provider == "openai":
        return OpenAIReasoning(thinking_level=thinking_level)
    if provider == "codex":
        return CodexReasoning(thinking_level=thinking_level)
    if provider == "google":
        if thinking_level == ThinkingLevel.BUDGET:
            return GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=GOOGLE_FIXED_BUDGET,
            )
        return GoogleReasoning(thinking_level=thinking_level)
    raise ValueError(f"unsupported provider: {provider!r}")


def format_attempt(
    provider: str,
    model: str,
    thinking_level: ThinkingLevel,
    effort: EffortSpec,
    reasoning_override: ReasoningSpec | None = None,
) -> str:
    if provider == "openrouter":
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
    if provider == "google" and thinking_level == ThinkingLevel.BUDGET:
        detail = (
            f"{provider} | {model} | "
            f"thinking={thinking_level.name}({GOOGLE_FIXED_BUDGET})"
        )
    if provider == "kimi-code" and thinking_level == ThinkingLevel.BUDGET:
        detail = (
            f"{provider} | {model} | "
            f"thinking={thinking_level.name}({KIMI_CODE_FIXED_BUDGET})"
        )
    if effort != EffortSpec.NA:
        detail += f" | effort={effort.name}"
    return detail


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
    effort: EffortSpec,
    *,
    explicit_reasoning: bool = False,
    reasoning_override: ReasoningSpec | None = None,
) -> LlmRequest:
    max_tokens = KIMI_CODE_MAX_TOKENS if provider == "kimi-code" else None
    reasoning = reasoning_override or reasoning_for_level(
        provider,
        thinking_level,
        explicit=explicit_reasoning,
    )
    if provider in {"codex", "claude-code"}:
        return HeadlessLlmRequest(
            provider=cast(HeadlessProviderName, provider),
            model=model,
            messages=[Message(role="user", content=PROMPT)],
            effort=effort,
            reasoning=reasoning,
        )
    if provider == "kimi-code":
        return KimiCodeLlmRequest(
            provider=cast(KimiCodeProviderName, provider),
            model=model,
            messages=[Message(role="user", content=PROMPT)],
            max_tokens=KIMI_CODE_MAX_TOKENS,
            effort=effort,
            reasoning=reasoning,
        )
    return ApiLlmRequest(
        provider=cast(ApiProviderName, provider),
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
    return provider == "minimax"


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
                effort=default_effort_for_model(provider, model),
                explicit_reasoning=requires_explicit_reasoning(provider),
                counts=counts,
                reasoning_override=default_reasoning_override(provider, model),
            )


def run_thinking_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[str],
) -> None:
    print("\n== thinking ==")
    for provider in providers:
        if provider == "openrouter":
            continue
        for model in PROVIDER_MODELS[provider]:
            for thinking_level in supported_thinking_levels(provider, model):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="thinking",
                    thinking_level=thinking_level,
                    effort=default_effort_for_model(provider, model),
                    explicit_reasoning=True,
                    counts=counts,
                )


def run_effort_sweep(
    registry: ProviderRegistry,
    counts: dict[tuple[str, str], SummaryCounts],
    providers: list[str],
) -> None:
    print("\n== effort ==")
    for provider in providers:
        if provider == "openrouter":
            continue
        for model in PROVIDER_MODELS[provider]:
            for effort in supported_effort_levels(provider=provider, model=model):
                run_attempt(
                    registry=registry,
                    provider=provider,
                    model=model,
                    phase="effort",
                    thinking_level=default_thinking_for_model(provider, model),
                    effort=effort,
                    explicit_reasoning=requires_explicit_reasoning(provider),
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
        run_effort_sweep(registry, counts, providers)
        print_summary(counts, providers)
    finally:
        registry.close()


if __name__ == "__main__":
    app()
