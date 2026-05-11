from __future__ import annotations

from enum import StrEnum
from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm.names import (
    EffortSpec,
    OpenRouterEffortLevel,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.impls.openai.families import OpenAIFamilies


class OpenRouterControlRequestStyle(StrEnum):
    NONE = "none"
    ENABLED_FLAG = "enabled_flag"
    EFFORT = "effort"


class OpenRouterModelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    request_style: OpenRouterControlRequestStyle
    supports_disable: bool
    allowed_efforts: tuple[OpenRouterEffortLevel, ...] = ()
    default_effort: OpenRouterEffortLevel | None = None
    default_enabled: bool | None = None
    verified: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_default_effort(self) -> OpenRouterModelPolicy:
        if self.default_effort is None:
            return self
        if self.request_style != OpenRouterControlRequestStyle.EFFORT:
            raise ValueError(
                "default_effort is only allowed for effort-style policies"
            )
        if self.default_effort in self.allowed_efforts:
            return self
        allowed = ", ".join(self.allowed_efforts)
        raise ValueError(
            f"default_effort={self.default_effort!r} is not in "
            f"allowed_efforts for model={self.model!r}: {allowed}"
        )


def _load_openrouter_policies() -> dict[str, OpenRouterModelPolicy]:
    raw = yaml.safe_load(
        files("dr_llm.llm.providers.impls.openrouter.data")
        .joinpath("model_policies.yml")
        .read_text(encoding="utf-8")
    )
    return {
        model: OpenRouterModelPolicy(model=model, **fields)
        for model, fields in raw.items()
    }


class OpenRouterFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    policies: dict[str, OpenRouterModelPolicy] = Field(
        default_factory=_load_openrouter_policies
    )
    openai_families: OpenAIFamilies = Field(default_factory=OpenAIFamilies)

    def policy_for_model(self, model: str) -> OpenRouterModelPolicy | None:
        return self.policies.get(model)

    def allowed_models(self) -> tuple[str, ...]:
        return tuple(self.policies)

    def control_mode(self, model: str) -> ControlMode:
        policy = self.policy_for_model(model)
        if policy is None:
            return ControlMode.UNSUPPORTED
        return control_mode_for_policy(policy.request_style)

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        del model
        return (ThinkingLevel.NA,)

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        del model
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        del model
        return ()

    def default_effort(self, model: str) -> EffortSpec:
        del model
        return EffortSpec.NA


OPENROUTER_FAMILIES = OpenRouterFamilies()


def control_mode_for_policy(
    request_style: OpenRouterControlRequestStyle,
) -> ControlMode:
    if request_style == OpenRouterControlRequestStyle.ENABLED_FLAG:
        return ControlMode.OPENROUTER_TOGGLE
    if request_style == OpenRouterControlRequestStyle.EFFORT:
        return ControlMode.OPENROUTER_EFFORT
    return ControlMode.UNSUPPORTED
