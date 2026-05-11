from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    GoogleReasoning,
    google_literal_to_thinking_level,
)
from dr_llm.llm.providers.core.authoring import (
    build_provider_config,
    validate_budget_range,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.google.capabilities import (
    reasoning_capabilities_for_google,
)

type _GoogleBudgetThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.ADAPTIVE,
    ThinkingLevel.BUDGET,
]
type _GoogleLevelThinkingLevel = Literal[
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
]


class _GoogleBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.GOOGLE] = ProviderName.GOOGLE
    model: str
    max_tokens: int | None = None
    sampling: SamplingControls | None = None

    def _expected_mode(self) -> ReasoningMode:
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_model_family(self) -> _GoogleBaseConfig:
        mode = _google_reasoning_mode(self.model)
        expected_mode = self._expected_mode()
        if mode != expected_mode:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"provider={self.provider!r} reasoning mode "
                f"{expected_mode.value!r}; got {mode.value!r} "
                f"for model={self.model!r}"
            )
        return self

    def _reasoning(self) -> GoogleReasoning | None:
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


class GoogleLegacyConfig(_GoogleBaseConfig):
    def _expected_mode(self) -> ReasoningMode:
        return ReasoningMode.UNSUPPORTED


class GoogleBudgetConfig(_GoogleBaseConfig):
    thinking_level: _GoogleBudgetThinkingLevel | None = None
    budget_tokens: int | None = None
    include_thoughts: bool | None = None

    def _expected_mode(self) -> ReasoningMode:
        return ReasoningMode.GOOGLE_BUDGET

    @model_validator(mode="after")
    def _validate_budget(self) -> GoogleBudgetConfig:
        if self.include_thoughts is not None and self.thinking_level is None:
            raise ValueError(
                "include_thoughts requires an explicit thinking_level"
            )
        if (
            self.thinking_level == ThinkingLevel.BUDGET
            and self.budget_tokens is None
        ):
            raise ValueError("GoogleBudgetConfig requires budget_tokens")
        if (
            self.thinking_level != ThinkingLevel.BUDGET
            and self.budget_tokens is not None
        ):
            raise ValueError("budget_tokens requires thinking_level='budget'")
        if self.budget_tokens is not None:
            capabilities = reasoning_capabilities_for_google(self.model)
            if (
                capabilities is None
                or capabilities.min_budget_tokens is None
                or capabilities.max_budget_tokens is None
            ):
                raise ValueError(
                    f"{type(self).__name__} budget thinking is not supported "
                    f"for model={self.model!r}"
                )
            validate_budget_range(
                provider=self.provider,
                model=self.model,
                budget_tokens=self.budget_tokens,
                min_tokens=capabilities.min_budget_tokens,
                max_tokens=capabilities.max_budget_tokens,
            )
        return self

    def _reasoning(self) -> GoogleReasoning | None:
        if self.thinking_level is None:
            return None
        return GoogleReasoning(
            thinking_level=self.thinking_level,
            budget_tokens=self.budget_tokens,
            include_thoughts=self.include_thoughts,
        )


class GoogleLevelConfig(_GoogleBaseConfig):
    thinking_level: _GoogleLevelThinkingLevel | None = None
    include_thoughts: bool | None = None

    def _expected_mode(self) -> ReasoningMode:
        return ReasoningMode.GOOGLE_LEVEL

    @model_validator(mode="after")
    def _validate_level(self) -> GoogleLevelConfig:
        if self.include_thoughts is not None and self.thinking_level is None:
            raise ValueError(
                "include_thoughts requires an explicit thinking_level"
            )
        if self.thinking_level is None:
            return self
        capabilities = reasoning_capabilities_for_google(self.model)
        if capabilities is None:
            raise ValueError(
                f"{type(self).__name__} thinking is not supported for "
                f"model={self.model!r}"
            )
        allowed_levels = {
            google_literal_to_thinking_level(level)
            for level in capabilities.google_thinking_levels
        }
        if self.thinking_level not in allowed_levels:
            allowed = ", ".join(sorted(str(level) for level in allowed_levels))
            raise ValueError(
                f"thinking_level={self.thinking_level.value!r} is not "
                f"supported for provider={self.provider!r} "
                f"model={self.model!r}; allowed levels: {allowed}"
            )
        return self

    def _reasoning(self) -> GoogleReasoning | None:
        if self.thinking_level is None:
            return None
        return GoogleReasoning(
            thinking_level=self.thinking_level,
            include_thoughts=self.include_thoughts,
        )


def _google_reasoning_mode(model: str) -> ReasoningMode:
    capabilities = reasoning_capabilities_for_google(model)
    if capabilities is None:
        return ReasoningMode.UNSUPPORTED
    return capabilities.mode
