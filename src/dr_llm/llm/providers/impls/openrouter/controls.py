from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    OpenRouterEffortLevel,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.openai.controls import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode

if TYPE_CHECKING:
    from dr_llm.llm.catalog.models import ModelCatalogEntry


class OpenRouterReasoningRequestStyle(StrEnum):
    NONE = "none"
    ENABLED_FLAG = "enabled_flag"
    EFFORT = "effort"


class OpenRouterModelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    request_style: OpenRouterReasoningRequestStyle
    supports_disable: bool
    allowed_efforts: tuple[OpenRouterEffortLevel, ...] = ()
    default_enabled: bool | None = None
    verified: bool = False
    notes: str | None = None


@cache
def _policies() -> dict[str, OpenRouterModelPolicy]:
    raw = yaml.safe_load(
        files("dr_llm.llm.providers.impls.openrouter.data")
        .joinpath("model_policies.yml")
        .read_text(encoding="utf-8")
    )
    return {
        model: OpenRouterModelPolicy(model=model, **fields)
        for model, fields in raw.items()
    }


def openrouter_model_policy(model: str) -> OpenRouterModelPolicy | None:
    return _policies().get(model)


def openrouter_reasoning_mode(model: str) -> ReasoningMode:
    policy = openrouter_model_policy(model)
    if policy is None:
        return ReasoningMode.UNSUPPORTED
    return _reasoning_mode_for_policy(policy.request_style)


def openrouter_allowed_models() -> tuple[str, ...]:
    return tuple(_policies())


def apply_openrouter_model_policies(
    entries: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    filtered: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.provider != ProviderName.OPENROUTER:
            filtered.append(entry)
            continue
        policy = openrouter_model_policy(entry.model)
        if policy is None:
            continue
        controls = OpenRouterControls(model=entry.model, mode=CallMode.api)
        filtered.append(
            entry.model_copy(
                update={
                    "supports_reasoning": controls.supports_reasoning,
                    "metadata": {
                        **entry.metadata,
                        "dr_llm_controls": controls.catalog_metadata,
                    },
                }
            )
        )
    return filtered


def _reasoning_mode_for_policy(
    request_style: OpenRouterReasoningRequestStyle,
) -> ReasoningMode:
    if request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        return ReasoningMode.OPENROUTER_TOGGLE
    if request_style == OpenRouterReasoningRequestStyle.EFFORT:
        return ReasoningMode.OPENROUTER_EFFORT
    return ReasoningMode.UNSUPPORTED


def validate_reasoning_for_openrouter(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if openrouter_model_policy(model) is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} model={model!r} is not in the curated allowlist"
        )
    reasoning_mode = openrouter_reasoning_mode(model)
    if reasoning is None:
        if not is_reasoning_unsupported(reasoning_mode):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.OPENROUTER}' model={model!r}"
            )
        return
    if isinstance(reasoning, OpenRouterReasoning):
        _validate_openrouter_shape(
            model=model,
            enabled=reasoning.enabled,
            effort=reasoning.effort,
        )
        return
    if isinstance(reasoning, OpenAIReasoning):
        if not openai_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENROUTER,
            model=model,
            thinking_level=reasoning.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.OPENROUTER}' model={model!r}; use OpenRouterReasoning or OpenAIReasoning"
        )
    raise ValueError(
        f"{ProviderName.OPENROUTER} reasoning is not supported for kind={reasoning.kind!r}"
    )


class OpenRouterControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.OPENROUTER
    model: str
    mode: CallMode

    @property
    def policy(self) -> OpenRouterModelPolicy | None:
        return openrouter_model_policy(self.model)

    @property
    def reasoning_mode(self) -> ReasoningMode:
        return openrouter_reasoning_mode(self.model)

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_mode != ReasoningMode.UNSUPPORTED

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        return (ThinkingLevel.NA,)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return ()

    @property
    def default_effort(self) -> EffortSpec:
        return EffortSpec.NA

    @property
    def default_reasoning(self) -> ReasoningSpec | None:
        policy = self.policy
        if policy is None:
            raise ValueError(
                f"missing openrouter policy for model={self.model!r}"
            )
        if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
            return None
        if (
            policy.request_style
            == OpenRouterReasoningRequestStyle.ENABLED_FLAG
        ):
            if policy.supports_disable:
                return OpenRouterReasoning(enabled=False)
            return OpenRouterReasoning(enabled=True)
        return OpenRouterReasoning(effort=policy.allowed_efforts[0])

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        policy = self.policy
        policy_metadata = policy.model_dump(mode="python") if policy else None
        return {
            "reasoning_mode": self.reasoning_mode,
            "supported_thinking_levels": self.supported_thinking_levels,
            "default_thinking_level": self.default_thinking_level,
            "supported_effort_levels": self.supported_effort_levels,
            "default_effort": self.default_effort,
            "policy": policy_metadata,
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
        raise ValueError(
            f"openrouter does not support thinking_level={thinking_level!r}"
        )

    def validate_request(self, request: LlmRequest) -> list:
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_openrouter(
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


def _validate_openrouter_shape(
    *,
    model: str,
    enabled: bool | None,
    effort: str | None,
) -> None:
    policy = openrouter_model_policy(model)
    if policy is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterReasoningRequestStyle.NONE:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        if effort is not None:
            raise ValueError(
                f"{ProviderName.OPENROUTER} effort controls are not supported for model={model!r}"
            )
        if enabled is None:
            raise ValueError(
                f"{ProviderName.OPENROUTER} reasoning requires the enabled flag for model={model!r}"
            )
        if not enabled and not policy.supports_disable:
            raise ValueError(
                f"{ProviderName.OPENROUTER} reasoning cannot be disabled for model={model!r}"
            )
        return
    if enabled is not None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} enabled controls are not supported for model={model!r}"
        )
    if effort is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning requires an effort level for model={model!r}"
        )
    if effort not in policy.allowed_efforts:
        allowed = ", ".join(policy.allowed_efforts)
        raise ValueError(
            f"{ProviderName.OPENROUTER} effort={effort!r} is not supported for model={model!r}; allowed levels: {allowed}"
        )


class OpenRouterReasoningConfig(BaseProviderReasoningConfig):
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high"] | None
    ) = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> OpenRouterReasoningConfig:
        if config is None:
            return cls()
        match config:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort="none")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort="minimal")
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort="low")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort="medium")
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort="high")
            case OpenRouterReasoning(enabled=enabled, effort=effort):
                reasoning_payload: dict[str, Any]
                if enabled is not None:
                    reasoning_payload = {"enabled": enabled}
                elif effort is not None:
                    reasoning_payload = {"effort": effort}
                else:
                    raise ProviderSemanticError(
                        "OpenRouter reasoning serializer received invalid config"
                    )
                return cls(extra_body={"reasoning": reasoning_payload})
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.OPENROUTER, config)
        )
