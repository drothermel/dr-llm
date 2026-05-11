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
from dr_llm.llm.providers.concepts.effort import validate_effort
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
from dr_llm.llm.providers.impls.codex.families import (
    CODEX_FAMILIES,
    CodexFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class CodexCliConfigKey(StrEnum):
    MODEL_REASONING_EFFORT = "model_reasoning_effort"


class CodexReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


def _validate_reasoning_for_codex(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: CodexFamilies | None = None,
) -> None:
    families = families or CODEX_FAMILIES

    def _validate_native(spec: CodexReasoning) -> None:
        if not families.supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.CODEX} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.CODEX,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=families.supports_off_thinking(model),
            supports_minimal=families.supports_minimal_thinking(model),
            supports_xhigh=True,
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        if is_control_unsupported(families.control_mode(model)):
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
        requires_reasoning=families.supports_configurable_thinking(model),
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
                return cls(
                    cli_args=[
                        "-c",
                        _codex_reasoning_effort_config(
                            CodexReasoningEffort.NONE
                        ),
                    ]
                )
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
                | ThinkingLevel.XHIGH
            ):
                return cls(
                    cli_args=[
                        "-c",
                        _codex_reasoning_effort_config(
                            CodexReasoningEffort(config.thinking_level)
                        ),
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
    families: CodexFamilies = Field(
        default_factory=CodexFamilies, exclude=True
    )

    @property
    def control_mode(self) -> ControlMode:
        return self.families.control_mode(self.model)

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        return self.families.supported_thinking_levels(self.model)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        return self.families.default_thinking_level(self.model)

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return self.families.supported_effort_levels(self.model)

    @property
    def default_effort(self) -> EffortSpec:
        return self.families.default_effort(self.model)

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
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_codex(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )
        if request.has_sampling_controls:
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
            )
        return []


def _codex_reasoning_effort_config(effort: CodexReasoningEffort) -> str:
    return f'{CodexCliConfigKey.MODEL_REASONING_EFFORT}="{effort}"'
