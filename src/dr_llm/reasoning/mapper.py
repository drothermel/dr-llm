from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.types import (
    CallMode,
    ReasoningConfig,
    ReasoningWarning,
    ReasoningWarningCode,
)


_EFFORT_RATIO = {
    "xhigh": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "minimal": 0.1,
    "none": 0.0,
}

_GOOGLE_THINKING_LEVEL = {
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "minimal",
    "none": "minimal",
}

_CLAUDE_SUPPORTED_EFFORT = {"low", "medium", "high"}


class ReasoningMappingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)


def map_reasoning_for_openai_compat(
    reasoning: ReasoningConfig | None,
    *,
    provider: str,
    mode: CallMode,
) -> ReasoningMappingResult:
    if reasoning is None:
        return ReasoningMappingResult()
    return ReasoningMappingResult(
        payload=reasoning.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        ),
    )


def map_reasoning_for_anthropic(
    reasoning: ReasoningConfig | None,
    *,
    provider: str,
    mode: CallMode,
    request_max_tokens: int | None,
) -> ReasoningMappingResult:
    if reasoning is None:
        return ReasoningMappingResult()
    warnings: list[ReasoningWarning] = []
    budget: int | None = None
    if reasoning.max_tokens is not None:
        budget = max(1024, min(int(reasoning.max_tokens), 128000))
    elif reasoning.effort is not None:
        max_tokens = int(request_max_tokens or 2048)
        ratio = _EFFORT_RATIO.get(reasoning.effort, 0.5)
        budget = max(1024, min(int(max_tokens * ratio), 128000))
    if budget is None:
        return ReasoningMappingResult()
    if request_max_tokens is not None and request_max_tokens <= 1024:
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.mapped_with_heuristic,
                message=(
                    "anthropic reasoning budget cannot be strictly less than max_tokens when max_tokens <= 1024; reasoning payload was omitted"
                ),
                provider=provider,
                mode=mode,
                details={"requested_max_tokens": request_max_tokens},
            )
        )
        return ReasoningMappingResult(warnings=warnings)
    if request_max_tokens is not None and budget >= request_max_tokens:
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.mapped_with_heuristic,
                message=(
                    "anthropic reasoning budget was clamped below max_tokens to leave room for final output"
                ),
                provider=provider,
                mode=mode,
                details={
                    "budget_tokens": max(1024, request_max_tokens - 1),
                    "requested_max_tokens": request_max_tokens,
                },
            )
        )
        budget = max(1024, request_max_tokens - 1)
    return ReasoningMappingResult(
        payload={"type": "enabled", "budget_tokens": budget},
        warnings=warnings,
    )


def map_reasoning_for_google(
    reasoning: ReasoningConfig | None,
    *,
    provider: str,
    mode: CallMode,
) -> ReasoningMappingResult:
    if reasoning is None:
        return ReasoningMappingResult()
    payload: dict[str, Any] = {}
    warnings: list[ReasoningWarning] = []
    if reasoning.max_tokens is not None:
        payload["thinkingBudget"] = int(reasoning.max_tokens)
    if reasoning.effort is not None:
        payload["thinkingLevel"] = _GOOGLE_THINKING_LEVEL.get(
            reasoning.effort, "medium"
        )
    if reasoning.exclude is True:
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.partially_supported,
                message="google reasoning.exclude is not directly supported and was ignored",
                provider=provider,
                mode=mode,
            )
        )
    if reasoning.enabled is False:
        payload["thinkingBudget"] = 0
        payload["thinkingLevel"] = "minimal"
    return ReasoningMappingResult(payload=payload, warnings=warnings)


def map_reasoning_for_claude_headless(
    reasoning: ReasoningConfig | None,
    *,
    provider: str,
    mode: CallMode,
) -> ReasoningMappingResult:
    if reasoning is None:
        return ReasoningMappingResult()
    warnings: list[ReasoningWarning] = []
    effort = reasoning.effort
    if effort is None and reasoning.max_tokens is not None:
        if reasoning.max_tokens >= 4000:
            effort = "high"
        elif reasoning.max_tokens >= 2000:
            effort = "medium"
        else:
            effort = "low"
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.mapped_with_heuristic,
                message=(
                    "mapped reasoning.max_tokens to claude --effort using heuristic thresholds"
                ),
                provider=provider,
                mode=mode,
                details={"derived_effort": effort, "max_tokens": reasoning.max_tokens},
            )
        )
    if effort == "xhigh":
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.mapped_with_heuristic,
                message="claude headless does not support xhigh effort; mapped to high",
                provider=provider,
                mode=mode,
            )
        )
        effort = "high"
    if effort == "minimal":
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.mapped_with_heuristic,
                message="claude headless does not support minimal effort; mapped to low",
                provider=provider,
                mode=mode,
            )
        )
        effort = "low"
    if effort == "none":
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.partially_supported,
                message="claude headless does not support disabling reasoning explicitly; using low effort",
                provider=provider,
                mode=mode,
            )
        )
        effort = "low"
    if effort is None:
        return ReasoningMappingResult(warnings=warnings)
    if effort not in _CLAUDE_SUPPORTED_EFFORT:
        warnings.append(
            ReasoningWarning(
                code=ReasoningWarningCode.unsupported_for_provider,
                message=f"claude headless effort {effort!r} is unsupported and was ignored",
                provider=provider,
                mode=mode,
            )
        )
        return ReasoningMappingResult(warnings=warnings)
    return ReasoningMappingResult(cli_args=["--effort", effort], warnings=warnings)


def map_reasoning_for_codex_headless(
    reasoning: ReasoningConfig | None,
    *,
    provider: str,
    mode: CallMode,
) -> ReasoningMappingResult:
    if reasoning is None:
        return ReasoningMappingResult()
    return ReasoningMappingResult(
        warnings=[
            ReasoningWarning(
                code=ReasoningWarningCode.unsupported_for_provider,
                message=(
                    "codex headless does not expose stable reasoning controls; request reasoning was not mapped"
                ),
                provider=provider,
                mode=mode,
                details=reasoning.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_computed_fields=True,
                ),
            )
        ]
    )
