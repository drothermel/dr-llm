from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
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
    GlmReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_control_unsupported,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.glm.families import (
    GLM_FAMILIES,
    GlmFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class GlmThinkingType(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"


GLM_DEFAULT_SAMPLING = SamplingControls(temperature=1.0, top_p=0.95)


def _validate_reasoning_for_glm(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: GlmFamilies | None = None,
) -> None:
    families = families or GLM_FAMILIES
    control_mode = families.control_mode(model)
    if reasoning is None:
        if not is_control_unsupported(control_mode):
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
    families: GlmFamilies = Field(default_factory=GlmFamilies, exclude=True)

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
            sampling_supported=True,
            sampling=GLM_DEFAULT_SAMPLING,
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
        return GLM_DEFAULT_SAMPLING

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
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_glm(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )
        return []


class GlmControlMapping(BaseProviderControlMapping):
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GlmControlMapping:
        if config is None:
            return cls()
        match config:
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(
                    extra_body={"thinking": {"type": GlmThinkingType.DISABLED}}
                )
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(
                    extra_body={"thinking": {"type": GlmThinkingType.ENABLED}}
                )
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.GLM, config)
        )
