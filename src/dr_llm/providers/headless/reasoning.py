from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.providers.models import CallMode, ReasoningWarning, ReasoningWarningCode
from dr_llm.providers.reasoning import ReasoningConfig


_CLAUDE_SUPPORTED_EFFORT = {"low", "medium", "high"}


class ClaudeHeadlessReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningConfig | None,
        *,
        provider: str,
        mode: CallMode,
    ) -> ClaudeHeadlessReasoningConfig:
        if config is None:
            return cls()
        warnings: list[ReasoningWarning] = []
        effort = config.effort
        if effort is None and config.max_tokens is not None:
            if config.max_tokens >= 4000:
                effort = "high"
            elif config.max_tokens >= 2000:
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
                    details={"derived_effort": effort, "max_tokens": config.max_tokens},
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
            return cls(warnings=warnings)
        if effort not in _CLAUDE_SUPPORTED_EFFORT:
            warnings.append(
                ReasoningWarning(
                    code=ReasoningWarningCode.unsupported_for_provider,
                    message=f"claude headless effort {effort!r} is unsupported and was ignored",
                    provider=provider,
                    mode=mode,
                )
            )
            return cls(warnings=warnings)
        return cls(cli_args=["--effort", effort], warnings=warnings)

    def to_cli_args(self) -> list[str]:
        return self.cli_args


class CodexHeadlessReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningConfig | None,
        *,
        provider: str,
        mode: CallMode,
    ) -> CodexHeadlessReasoningConfig:
        if config is None:
            return cls()
        return cls(
            warnings=[
                ReasoningWarning(
                    code=ReasoningWarningCode.unsupported_for_provider,
                    message=(
                        "codex headless does not expose stable reasoning controls; request reasoning was not mapped"
                    ),
                    provider=provider,
                    mode=mode,
                    details=config.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    ),
                )
            ]
        )

    def to_cli_args(self) -> list[str]:
        return self.cli_args
