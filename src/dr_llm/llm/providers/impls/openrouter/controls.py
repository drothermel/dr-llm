from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.effort import validate_effort
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ReasoningSpec,
    is_control_unsupported,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.openrouter.families import (
    OPENROUTER_FAMILIES,
    OpenRouterControlRequestStyle,
    OpenRouterFamilies,
    OpenRouterModelPolicy,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode

if TYPE_CHECKING:
    from dr_llm.llm.catalog.models import ModelCatalogEntry


OPENROUTER_DEFAULT_SAMPLING = SamplingControls(temperature=1.0, top_p=0.95)


def _default_openrouter_families() -> OpenRouterFamilies:
    return OPENROUTER_FAMILIES


def apply_openrouter_model_policies(
    entries: list[ModelCatalogEntry],
    families: OpenRouterFamilies | None = None,
) -> list[ModelCatalogEntry]:
    families = families or OPENROUTER_FAMILIES
    filtered: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.provider != ProviderName.OPENROUTER:
            filtered.append(entry)
            continue
        policy = families.policy_for_model(entry.model)
        if policy is None:
            continue
        controls = OpenRouterControls(
            model=entry.model, mode=CallMode.api, families=families
        )
        filtered.append(
            entry.model_copy(
                update={
                    "control_mode": controls.control_mode,
                    "metadata": {
                        **entry.metadata,
                        "dr_llm_controls": controls.catalog_metadata,
                    },
                }
            )
        )
    return filtered


def _validate_reasoning_for_openrouter(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: OpenRouterFamilies | None = None,
) -> None:
    families = families or OPENROUTER_FAMILIES
    if families.policy_for_model(model) is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} model={model!r} is not in the curated allowlist"
        )
    control_mode = families.control_mode(model)
    if reasoning is None:
        if not is_control_unsupported(control_mode):
            raise ValueError(
                f"reasoning is required for provider='{ProviderName.OPENROUTER}' model={model!r}"
            )
        return
    if isinstance(reasoning, OpenRouterReasoning):
        _validate_openrouter_shape(
            model=model,
            enabled=reasoning.enabled,
            effort=reasoning.effort,
            families=families,
        )
        return
    if isinstance(reasoning, OpenAIReasoning):
        if not families.openai_families.supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENROUTER,
            model=model,
            thinking_level=reasoning.thinking_level,
            supports_off=families.openai_families.supports_off_thinking(model),
            supports_minimal=(
                families.openai_families.supports_minimal_thinking(model)
            ),
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
    families: OpenRouterFamilies = Field(
        default_factory=_default_openrouter_families, exclude=True
    )

    @property
    def policy(self) -> OpenRouterModelPolicy | None:
        return self.families.policy_for_model(self.model)

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
        policy = self.policy
        if policy is None:
            raise ValueError(
                f"missing openrouter policy for model={self.model!r}"
            )
        if policy.request_style == OpenRouterControlRequestStyle.NONE:
            return None
        if policy.request_style == OpenRouterControlRequestStyle.ENABLED_FLAG:
            if policy.supports_disable:
                return OpenRouterReasoning(enabled=False)
            return OpenRouterReasoning(enabled=True)
        return OpenRouterReasoning(effort=policy.allowed_efforts[0])

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        policy = self.policy
        policy_metadata = policy.model_dump(mode="python") if policy else None
        return {
            "control_mode": self.control_mode,
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
            sampling=OPENROUTER_DEFAULT_SAMPLING,
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
        return OPENROUTER_DEFAULT_SAMPLING

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
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_openrouter(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )
        return []


def _validate_openrouter_shape(
    *,
    model: str,
    enabled: bool | None,
    effort: str | None,
    families: OpenRouterFamilies,
) -> None:
    policy = families.policy_for_model(model)
    if policy is None:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterControlRequestStyle.NONE:
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning is not supported for model={model!r}"
        )
    if policy.request_style == OpenRouterControlRequestStyle.ENABLED_FLAG:
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
