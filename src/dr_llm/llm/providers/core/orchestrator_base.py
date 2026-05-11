from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ThinkingLevel,
)
from dr_llm.llm.providers.core.base import ProviderTransport
from dr_llm.llm.providers.concepts.capabilities import ModelCapabilities
from dr_llm.llm.providers.concepts.reasoning import (
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.core.reasoning_controls import ReasoningControls
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest, Message, parse_llm_request
from dr_llm.llm.response import CallMode, LlmResponse

CatalogResult = tuple[list[ModelCatalogEntry], dict[str, Any]]


class BaseProviderOrchestrator(ABC):
    def __init__(self, provider: ProviderTransport) -> None:
        self._provider = provider

    @property
    def name(self) -> ProviderName:
        return ProviderName(self._provider.name)

    @property
    def mode(self) -> CallMode:
        return self._provider.mode

    def availability_status(self) -> ProviderAvailabilityStatus:
        return self._provider.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    def close(self) -> None:
        self._provider.close()

    def reasoning_controls(self, model: str) -> ReasoningControls:
        capabilities = self.model_capabilities(model)
        supported_levels = self._supported_thinking_levels(
            model=model, capabilities=capabilities
        )
        default_level = self.default_thinking_level(model, supported_levels)
        return ReasoningControls(
            provider=self.name,
            model=model,
            supported_thinking_levels=supported_levels,
            default_thinking_level=default_level,
            supported_effort_levels=capabilities.supported_effort_levels,
            default_effort=self.default_effort(capabilities),
            default_reasoning=self.default_reasoning(
                model=model,
                thinking_level=default_level,
                capabilities=capabilities,
            ),
        )

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        controls = self.reasoning_controls(model)
        return ProviderRequestDefaults(
            provider=self.name,
            model=model,
            mode=self.mode,
            effort=controls.default_effort,
            reasoning=controls.default_reasoning,
        )

    def build_request(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        effort: EffortSpec = EffortSpec.NA,
        reasoning: ReasoningSpec | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        defaults = self.request_defaults(model)
        payload: dict[str, Any] = {
            "provider": defaults.provider,
            "model": defaults.model,
            "messages": messages,
            "effort": self._resolve_effort(defaults, effort),
            "reasoning": defaults.reasoning
            if reasoning is None
            else reasoning,
            "metadata": metadata or {},
        }

        resolved_max_tokens = (
            defaults.max_tokens if max_tokens is None else max_tokens
        )
        if resolved_max_tokens is not None:
            if defaults.mode == CallMode.headless:
                raise ValueError(
                    f"max_tokens is not supported for provider={self.name!r}"
                )
            payload["max_tokens"] = resolved_max_tokens

        if temperature is not None and not defaults.supports_temperature:
            raise ValueError(
                f"temperature is not supported for provider={self.name!r}"
            )
        resolved_temperature = (
            defaults.temperature if temperature is None else temperature
        )
        if defaults.supports_temperature:
            payload["temperature"] = resolved_temperature

        if top_p is not None and not defaults.supports_top_p:
            raise ValueError(
                f"top_p is not supported for provider={self.name!r}"
            )
        resolved_top_p = defaults.top_p if top_p is None else top_p
        if defaults.supports_top_p:
            payload["top_p"] = resolved_top_p

        request = parse_llm_request(payload)
        self.validate_request(request)
        return request

    def _resolve_effort(
        self, defaults: ProviderRequestDefaults, effort: EffortSpec
    ) -> EffortSpec:
        if effort == EffortSpec.NA and defaults.effort != EffortSpec.NA:
            return defaults.effort
        return effort

    @abstractmethod
    def model_capabilities(self, model: str) -> ModelCapabilities: ...

    @abstractmethod
    def fetch_models(self) -> CatalogResult: ...

    def fallback_models(self) -> CatalogResult:
        return [], {"source": "static"}

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
        return self._supported_thinking_levels(
            model=model, capabilities=self.model_capabilities(model)
        )

    @abstractmethod
    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]: ...

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

    def default_effort(self, capabilities: ModelCapabilities) -> EffortSpec:
        effort_levels = capabilities.supported_effort_levels
        if effort_levels:
            return effort_levels[0]
        return EffortSpec.NA

    def default_reasoning(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> ReasoningSpec | None:
        level = thinking_level or self.default_thinking_level(model)
        resolved_capabilities = capabilities or self.model_capabilities(model)
        budget_tokens = resolved_capabilities.reasoning.min_budget_tokens
        return self.reasoning_for_thinking_level(
            model=model,
            thinking_level=level,
            budget_tokens=budget_tokens,
        )

    @abstractmethod
    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None: ...

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

    def _require_budget_tokens(self, budget_tokens: int | None) -> int:
        if budget_tokens is None:
            raise ValueError(
                f"{self.name} budget thinking requires budget_tokens"
            )
        return budget_tokens
