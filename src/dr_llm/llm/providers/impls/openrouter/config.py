from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import OpenRouterEffortLevel, ProviderName
from dr_llm.llm.providers.concepts.reasoning import OpenRouterReasoning
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)


class _OpenRouterBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.OPENROUTER] = ProviderName.OPENROUTER
    model: str
    max_tokens: int | None = None
    sampling: SamplingControls | None = None

    def _expected_style(self) -> OpenRouterReasoningRequestStyle:
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_policy(self) -> _OpenRouterBaseConfig:
        policy = openrouter_model_policy(self.model)
        if policy is None:
            raise ValueError(
                f"{ProviderName.OPENROUTER} model={self.model!r} is not in "
                "the curated allowlist"
            )
        expected_style = self._expected_style()
        if policy.request_style != expected_style:
            raise ValueError(
                f"{type(self).__name__} requires openrouter request_style "
                f"{expected_style.value!r}; got "
                f"{policy.request_style.value!r} for model={self.model!r}"
            )
        return self

    def _reasoning(self) -> OpenRouterReasoning | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning=self._reasoning(),
            sampling=self.sampling,
            registry=registry,
        )


class OpenRouterNoReasoningConfig(_OpenRouterBaseConfig):
    def _expected_style(self) -> OpenRouterReasoningRequestStyle:
        return OpenRouterReasoningRequestStyle.NONE


class OpenRouterToggleConfig(_OpenRouterBaseConfig):
    enabled: bool | None = None

    def _expected_style(self) -> OpenRouterReasoningRequestStyle:
        return OpenRouterReasoningRequestStyle.ENABLED_FLAG

    @model_validator(mode="after")
    def _validate_enabled(self) -> OpenRouterToggleConfig:
        if self.enabled is not False:
            return self
        policy = openrouter_model_policy(self.model)
        if policy is not None and policy.supports_disable:
            return self
        raise ValueError(
            f"{ProviderName.OPENROUTER} reasoning cannot be disabled for "
            f"model={self.model!r}"
        )

    def _reasoning(self) -> OpenRouterReasoning | None:
        if self.enabled is None:
            return None
        return OpenRouterReasoning(enabled=self.enabled)


class OpenRouterEffortConfig(_OpenRouterBaseConfig):
    effort: OpenRouterEffortLevel | None = None

    def _expected_style(self) -> OpenRouterReasoningRequestStyle:
        return OpenRouterReasoningRequestStyle.EFFORT

    @model_validator(mode="after")
    def _validate_effort(self) -> OpenRouterEffortConfig:
        if self.effort is None:
            return self
        policy = openrouter_model_policy(self.model)
        if policy is not None and self.effort in policy.allowed_efforts:
            return self
        allowed = (
            ", ".join(policy.allowed_efforts)
            if policy is not None
            else "<unknown>"
        )
        raise ValueError(
            f"{ProviderName.OPENROUTER} effort={self.effort!r} is not "
            f"supported for model={self.model!r}; allowed levels: {allowed}"
        )

    def _reasoning(self) -> OpenRouterReasoning | None:
        if self.effort is None:
            return None
        return OpenRouterReasoning(effort=self.effort)
