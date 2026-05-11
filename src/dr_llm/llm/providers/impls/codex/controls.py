from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.model_family import (
    is_snapshot_of_family,
    model_matches_any_family,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderControlMapping,
    CodexReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    is_control_unsupported,
    unsupported_reasoning_kind_message,
    validate_budget_range,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class CodexModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT51 = "gpt-5.1"
    GPT52 = "gpt-5.2"
    GPT54 = "gpt-5.4"
    GPT5_CODEX = "gpt-5-codex"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT53_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT54_MINI = "gpt-5.4-mini"

    def in_family(self, model: str) -> bool:
        return model == self or is_snapshot_of_family(
            model=model, family=str(self)
        )


CODEX_THINKING_SUPPORTED_MODELS = (
    CodexModelFamily.GPT5,
    CodexModelFamily.GPT51,
    CodexModelFamily.GPT52,
    CodexModelFamily.GPT54,
    CodexModelFamily.GPT5_CODEX,
    CodexModelFamily.GPT51_CODEX,
    CodexModelFamily.GPT51_CODEX_MINI,
    CodexModelFamily.GPT51_CODEX_MAX,
    CodexModelFamily.GPT52_CODEX,
    CodexModelFamily.GPT53_CODEX,
    CodexModelFamily.GPT53_CODEX_SPARK,
    CodexModelFamily.GPT54_MINI,
)
CODEX_MINIMAL_THINKING_SUPPORTED_MODELS = (CodexModelFamily.GPT5,)
CODEX_OFF_THINKING_SUPPORTED_MODELS = (
    CodexModelFamily.GPT51,
    CodexModelFamily.GPT52,
    CodexModelFamily.GPT54,
    CodexModelFamily.GPT54_MINI,
)


def codex_supports_configurable_thinking(model: str) -> bool:
    return model_matches_any_family(model, CODEX_THINKING_SUPPORTED_MODELS)


def codex_supports_minimal_thinking(model: str) -> bool:
    return model_matches_any_family(
        model, CODEX_MINIMAL_THINKING_SUPPORTED_MODELS
    )


def codex_supports_off_thinking(model: str) -> bool:
    return model_matches_any_family(model, CODEX_OFF_THINKING_SUPPORTED_MODELS)


def codex_control_mode(model: str) -> ControlMode:
    if codex_supports_configurable_thinking(model):
        return ControlMode.CODEX_CLI_EFFORT
    return ControlMode.UNSUPPORTED


def validate_reasoning_for_codex(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: CodexReasoning) -> None:
        if not codex_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.CODEX} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.CODEX,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=codex_supports_off_thinking(model),
            supports_minimal=codex_supports_minimal_thinking(model),
            supports_xhigh=True,
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        if is_control_unsupported(codex_control_mode(model)):
            raise ValueError(
                f"Reasoning is not supported for provider='{ProviderName.CODEX}' model={model!r}"
            )
        validate_budget_range(
            provider=ProviderName.CODEX,
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            min_budget_tokens=None,
            max_budget_tokens=None,
        )

    dispatch_reasoning_validation(
        provider=ProviderName.CODEX,
        model=model,
        reasoning=reasoning,
        native_spec_type=CodexReasoning,
        requires_reasoning=codex_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


class CodexHeadlessControlMapping(BaseProviderControlMapping):
    cli_args: list[str] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> CodexHeadlessControlMapping:
        if config is None:
            return cls()
        match config:
            case CodexReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case CodexReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(cli_args=["-c", 'model_reasoning_effort="none"'])
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
                | ThinkingLevel.XHIGH
            ):
                thinking_level = config.thinking_level
                return cls(
                    cli_args=[
                        "-c",
                        f'model_reasoning_effort="{thinking_level}"',
                    ]
                )
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("codex headless", config)
        )


class CodexControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.CODEX
    model: str
    mode: CallMode

    @property
    def control_mode(self) -> ControlMode:
        return codex_control_mode(self.model)

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if not codex_supports_configurable_thinking(self.model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if codex_supports_off_thinking(self.model):
            levels.append(ThinkingLevel.OFF)
        elif codex_supports_minimal_thinking(self.model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [
                ThinkingLevel.LOW,
                ThinkingLevel.MEDIUM,
                ThinkingLevel.HIGH,
                ThinkingLevel.XHIGH,
            ]
        )
        return tuple(levels)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        levels = self.supported_thinking_levels
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return ()

    @property
    def default_effort(self) -> EffortSpec:
        return EffortSpec.NA

    @property
    def default_reasoning(self) -> ReasoningSpec | None:
        return self.reasoning_for_thinking_level(
            thinking_level=self.default_thinking_level
        )

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        return {
            "control_mode": self.control_mode,
            "supported_thinking_levels": self.supported_thinking_levels,
            "default_thinking_level": self.default_thinking_level,
            "supported_effort_levels": self.supported_effort_levels,
            "default_effort": self.default_effort,
        }

    def request_defaults(self) -> ProviderRequestDefaults:
        return ProviderRequestDefaults(
            provider=self.provider,
            model=self.model,
            mode=self.mode,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
        )

    def resolve_reasoning(
        self,
        *,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if reasoning is not None:
            return reasoning
        if thinking_level is not None:
            return self.reasoning_for_thinking_level(
                thinking_level=thinking_level
            )
        return self.default_reasoning

    def resolve_effort(self, effort: EffortSpec | None) -> EffortSpec:
        if effort is None:
            return self.default_effort
        return effort

    def resolve_sampling(
        self, sampling: SamplingControls | None
    ) -> SamplingControls | None:
        if sampling is not None and not sampling.is_empty():
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
            )
        return None

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return None
        return CodexReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list:
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_codex(
            model=request.model, reasoning=request.reasoning
        )
        if request.has_sampling_controls:
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
            )
        return []


def _validate_effort(
    *,
    provider: str,
    model: str,
    effort: EffortSpec,
    supported_effort_levels: tuple[EffortSpec, ...],
) -> None:
    if not supported_effort_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} "
                f"model={model!r}"
            )
        return
    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )
    if effort not in supported_effort_levels:
        allowed = ", ".join(str(level) for level in supported_effort_levels)
        raise ValueError(
            f"effort={effort!r} is not supported for provider={provider!r} "
            f"model={model!r}; allowed levels: {allowed}"
        )
