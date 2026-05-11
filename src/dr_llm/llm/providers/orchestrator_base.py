from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.providers.concepts.capabilities import ModelCapabilities
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    GlmReasoning,
    GoogleReasoning,
    ReasoningSpec,
    ReasoningWarning,
    google_literal_to_thinking_level,
)
from dr_llm.llm.providers.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.reasoning_controls import ReasoningControls
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse

CatalogResult = tuple[list[ModelCatalogEntry], dict[str, Any]]


class BaseProviderOrchestrator(ABC):
    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    @property
    def name(self) -> ProviderName:
        return ProviderName(self._provider.name)

    @property
    def mode(self) -> str:
        return self._provider.mode

    def availability_status(self) -> ProviderAvailabilityStatus:
        return self._provider.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    def close(self) -> None:
        self._provider.close()

    def reasoning_controls(self, model: str) -> ReasoningControls:
        supported_levels = self.supported_thinking_levels(model)
        default_level = self.default_thinking_level(model, supported_levels)
        return ReasoningControls(
            provider=self.name,
            model=model,
            supported_thinking_levels=supported_levels,
            default_thinking_level=default_level,
            supported_effort_levels=self.model_capabilities(
                model
            ).supported_effort_levels,
            default_effort=self.default_effort(model),
            default_reasoning=self.default_reasoning(
                model=model, thinking_level=default_level
            ),
        )

    @abstractmethod
    def model_capabilities(self, model: str) -> ModelCapabilities: ...

    @abstractmethod
    def fetch_models(self) -> CatalogResult: ...

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        self._validate_provider(request)
        self._validate_effort(request)
        return []

    def generate(self, request: LlmRequest) -> LlmResponse:
        warnings = self.validate_request(request)
        response = self._provider.generate(request)
        if not warnings:
            return response
        return response.model_copy(
            update={"warnings": [*response.warnings, *warnings]}
        )

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        capabilities = self.model_capabilities(model).reasoning
        if capabilities.mode == ReasoningMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if capabilities.mode == ReasoningMode.GOOGLE_BUDGET:
            return (
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.OFF,
                ThinkingLevel.BUDGET,
            )
        if capabilities.mode == ReasoningMode.GOOGLE_LEVEL:
            return tuple(
                google_literal_to_thinking_level(level)
                for level in capabilities.google_levels
            )
        if capabilities.mode == ReasoningMode.GLM:
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        if capabilities.mode == ReasoningMode.ANTHROPIC_BUDGET:
            return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
        if capabilities.mode == ReasoningMode.ANTHROPIC_EFFORT:
            return (ThinkingLevel.OFF,)
        if capabilities.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET:
            return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
        if capabilities.mode == ReasoningMode.KIMI_CODE_EFFORT_AND_BUDGET:
            return (
                ThinkingLevel.OFF,
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.BUDGET,
            )
        if capabilities.mode == ReasoningMode.MINIMAX_EFFORT:
            return (ThinkingLevel.NA,)
        raise ValueError(
            f"unexpected reasoning mode for provider={self.name!r} "
            f"model={model!r}: {capabilities.mode!r}"
        )

    def default_thinking_level(
        self,
        model: str,
        supported_levels: tuple[ThinkingLevel, ...] | None = None,
    ) -> ThinkingLevel:
        del model
        levels = supported_levels or (ThinkingLevel.NA,)
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW,
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.BUDGET,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    def default_effort(self, model: str) -> EffortSpec:
        effort_levels = self.model_capabilities(model).supported_effort_levels
        if effort_levels:
            return effort_levels[0]
        return EffortSpec.NA

    def default_reasoning(
        self, *, model: str, thinking_level: ThinkingLevel | None = None
    ) -> ReasoningSpec | None:
        level = thinking_level or self.default_thinking_level(model)
        budget_tokens = self.model_capabilities(
            model
        ).reasoning.min_budget_tokens
        return self.reasoning_for_thinking_level(
            model=model,
            thinking_level=level,
            budget_tokens=budget_tokens,
        )

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del model
        if self.name == ProviderName.GOOGLE:
            return self._google_reasoning_for_thinking_level(
                thinking_level=thinking_level, budget_tokens=budget_tokens
            )
        if self.name == ProviderName.GLM:
            if thinking_level == ThinkingLevel.NA:
                return None
            return GlmReasoning(thinking_level=thinking_level)
        if self.name in {ProviderName.ANTHROPIC, ProviderName.KIMI_CODE}:
            return self._anthropic_reasoning_for_thinking_level(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                explicit_na=False,
            )
        if self.name == ProviderName.MINIMAX:
            return self._anthropic_reasoning_for_thinking_level(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                explicit_na=True,
            )
        raise ValueError(f"unsupported provider: {self.name!r}")

    def _validate_provider(self, request: LlmRequest) -> None:
        if request.provider != self.name:
            raise ValueError(
                f"request provider {request.provider!r} does not match "
                f"orchestrator provider {self.name!r}"
            )

    def _validate_effort(self, request: LlmRequest) -> None:
        allowed_levels = self.model_capabilities(
            request.model
        ).supported_effort_levels
        if not allowed_levels:
            if request.effort != EffortSpec.NA:
                raise ValueError(
                    f"effort is not supported for provider={self.name!r} "
                    f"model={request.model!r}"
                )
            return
        if request.effort == EffortSpec.NA:
            raise ValueError(
                f"effort is required for provider={self.name!r} "
                f"model={request.model!r}"
            )
        if request.effort not in allowed_levels:
            allowed = ", ".join(level.value for level in allowed_levels)
            raise ValueError(
                f"effort={request.effort.value!r} is not supported for "
                f"provider={self.name!r} model={request.model!r}; "
                f"allowed levels: {allowed}"
            )

    def _validate_max_tokens_required(self, request: LlmRequest) -> None:
        max_tokens = getattr(request, "max_tokens", None)
        if max_tokens is None:
            raise ValueError(
                f"max_tokens is required for provider={self.name!r}"
            )

    def _google_reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None,
    ) -> GoogleReasoning | None:
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=self._require_budget_tokens(budget_tokens),
            )
        return GoogleReasoning(thinking_level=thinking_level)

    def _anthropic_reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None,
        explicit_na: bool,
    ) -> AnthropicReasoning | None:
        if thinking_level == ThinkingLevel.NA:
            if explicit_na:
                return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=self._require_budget_tokens(budget_tokens),
            )
        return AnthropicReasoning(thinking_level=thinking_level)

    def _require_budget_tokens(self, budget_tokens: int | None) -> int:
        if budget_tokens is None:
            raise ValueError(
                f"{self.name} budget thinking requires budget_tokens"
            )
        return budget_tokens
