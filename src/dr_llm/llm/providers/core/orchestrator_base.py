from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.config import LlmConfig, SamplingControls
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
        effort: EffortSpec | None = None,
        reasoning: ReasoningSpec | None = None,
        thinking_level: ThinkingLevel | None = None,
        budget_tokens: int | None = None,
        sampling: SamplingControls | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        config = self.build_config(
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            reasoning=reasoning,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            sampling=sampling,
        )
        return self.build_request_from_config(
            config=config, messages=messages, metadata=metadata
        )

    def build_config(
        self,
        *,
        model: str,
        max_tokens: int | None = None,
        effort: EffortSpec | None = None,
        reasoning: ReasoningSpec | None = None,
        thinking_level: ThinkingLevel | None = None,
        budget_tokens: int | None = None,
        sampling: SamplingControls | None = None,
    ) -> LlmConfig:
        if reasoning is not None and thinking_level is not None:
            raise ValueError(
                "reasoning and thinking_level are mutually exclusive"
            )
        defaults = self.request_defaults(model)
        resolved_reasoning = self._resolve_reasoning(
            model=model,
            defaults=defaults,
            reasoning=reasoning,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
        )
        resolved_effort = (
            defaults.effort
            if effort is None
            else self._resolve_effort(defaults, effort)
        )
        payload: dict[str, Any] = {
            "provider": defaults.provider,
            "model": defaults.model,
            "mode": defaults.mode,
            "effort": resolved_effort,
            "reasoning": resolved_reasoning,
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

        resolved_sampling = self._resolve_sampling(defaults, sampling)
        if resolved_sampling is not None:
            payload["sampling"] = resolved_sampling

        config = LlmConfig(**payload)
        self.validate_config(config)
        return config

    def build_request_from_config(
        self,
        *,
        config: LlmConfig,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        self.validate_config(config)
        request = parse_llm_request(
            {
                **config.model_dump(mode="python"),
                "messages": messages,
                "metadata": metadata or {},
            }
        )
        self.validate_request(request)
        return request

    def validate_config(self, config: LlmConfig) -> None:
        if config.provider != self.name:
            raise ValueError(
                f"config provider {config.provider!r} does not match "
                f"orchestrator provider {self.name!r}"
            )
        if config.mode != self.mode:
            raise ValueError(
                f"config mode {config.mode!r} does not match "
                f"orchestrator mode {self.mode!r}"
            )
        self.validate_request(self._config_validation_request(config))

    def _config_validation_request(self, config: LlmConfig) -> LlmRequest:
        return parse_llm_request(
            {
                **config.model_dump(mode="python"),
                "messages": [],
                "metadata": {},
            }
        )

    def _resolve_reasoning(
        self,
        *,
        model: str,
        defaults: ProviderRequestDefaults,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None:
        if reasoning is not None:
            return reasoning
        if thinking_level is not None:
            return self.reasoning_for_thinking_level(
                model=model,
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
            )
        return defaults.reasoning

    def _resolve_effort(
        self, defaults: ProviderRequestDefaults, effort: EffortSpec
    ) -> EffortSpec:
        if effort == EffortSpec.NA and defaults.effort != EffortSpec.NA:
            return defaults.effort
        return effort

    def _resolve_sampling(
        self,
        defaults: ProviderRequestDefaults,
        sampling: SamplingControls | None,
    ) -> SamplingControls | None:
        if sampling is not None:
            if sampling.is_empty():
                return None
            if not defaults.sampling_supported:
                raise ValueError(
                    f"sampling is not supported for provider={self.name!r}"
                )
            return sampling
        if not defaults.sampling_supported or defaults.sampling is None:
            return None
        if defaults.sampling.is_empty():
            return None
        return defaults.sampling

    @abstractmethod
    def model_capabilities(self, model: str) -> ModelCapabilities: ...

    @abstractmethod
    def fetch_models(self) -> CatalogResult: ...

    def fallback_models(self) -> CatalogResult:
        return [], {"source": "static"}

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        self._validate_provider(request)
        self._validate_mode(request)
        self._validate_supported_request_controls(request)
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

    def _validate_mode(self, request: LlmRequest) -> None:
        if request.mode != self.mode:
            raise ValueError(
                f"request mode {request.mode!r} does not match "
                f"orchestrator mode {self.mode!r}"
            )

    def _validate_supported_request_controls(
        self, request: LlmRequest
    ) -> None:
        if request.max_tokens is not None and self.mode == CallMode.headless:
            raise ValueError(
                f"max_tokens is not supported for provider={self.name!r}"
            )
        if not request.has_sampling_controls:
            return
        defaults = self.request_defaults(request.model)
        if not defaults.sampling_supported:
            raise ValueError(
                f"sampling is not supported for provider={self.name!r}"
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
            allowed = ", ".join(str(level) for level in allowed_levels)
            raise ValueError(
                f"effort={request.effort!r} is not supported for "
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
