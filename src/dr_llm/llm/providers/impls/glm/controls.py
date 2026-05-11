from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    GlmReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class GlmModelFamily(StrEnum):
    GLM5 = "glm-5"
    GLM47 = "glm-4.7"
    GLM46 = "glm-4.6"
    GLM45 = "glm-4.5"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


GLM_THINKING_SUPPORTED_FAMILIES = (
    GlmModelFamily.GLM5,
    GlmModelFamily.GLM47,
    GlmModelFamily.GLM46,
    GlmModelFamily.GLM45,
)


def glm_reasoning_mode(model: str) -> ReasoningMode:
    if any(
        family.in_family(model) for family in GLM_THINKING_SUPPORTED_FAMILIES
    ):
        return ReasoningMode.GLM
    return ReasoningMode.UNSUPPORTED


def validate_reasoning_for_glm(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    reasoning_mode = glm_reasoning_mode(model)
    if reasoning is None:
        if not is_reasoning_unsupported(reasoning_mode):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.GLM}' model={model!r}"
            )
        return
    if isinstance(reasoning, GlmReasoning):
        validate_allowed_thinking_levels(
            provider=ProviderName.GLM,
            model=model,
            thinking_level=reasoning.thinking_level,
            allowed_levels={ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE},
            allow_na=False,
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.GLM}' model={model!r}; use GlmReasoning(thinking_level=...)"
        )
    raise ValueError(
        f"{ProviderName.GLM} reasoning is not supported for kind={reasoning.kind!r}"
    )


class GlmControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.GLM
    model: str
    mode: CallMode

    @property
    def reasoning_mode(self) -> ReasoningMode:
        return glm_reasoning_mode(self.model)

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_mode != ReasoningMode.UNSUPPORTED

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if self.reasoning_mode == ReasoningMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if self.reasoning_mode == ReasoningMode.GLM:
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        raise ValueError(
            f"unexpected reasoning mode for provider={self.provider!r} "
            f"model={self.model!r}: {self.reasoning_mode!r}"
        )

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        if ThinkingLevel.OFF in self.supported_thinking_levels:
            return ThinkingLevel.OFF
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
            "reasoning_mode": self.reasoning_mode,
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
            sampling_supported=True,
            sampling=SamplingControls(temperature=1.0, top_p=0.95),
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
        if sampling is not None:
            if sampling.is_empty():
                return None
            return sampling
        return SamplingControls(temperature=1.0, top_p=0.95)

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return None
        return GlmReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list:
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_glm(
            model=request.model, reasoning=request.reasoning
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


class GlmReasoningConfig(BaseProviderReasoningConfig):
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GlmReasoningConfig:
        if config is None:
            return cls()
        match config:
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(extra_body={"thinking": {"type": "disabled"}})
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(extra_body={"thinking": {"type": "enabled"}})
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.GLM, config)
        )
