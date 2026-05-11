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
from dr_llm.llm.providers.concepts.reasoning import (
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.core.controls import ProviderControls
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

    @abstractmethod
    def controls(self, model: str) -> ProviderControls: ...

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        return self.controls(model).request_defaults()

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
        controls = self.controls(model)
        defaults = controls.request_defaults()
        resolved_reasoning = controls.resolve_reasoning(
            reasoning=reasoning,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
        )
        resolved_effort = controls.resolve_effort(effort)
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

        resolved_sampling = controls.resolve_sampling(sampling)
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

    @abstractmethod
    def fetch_models(self) -> CatalogResult: ...

    def fallback_models(self) -> CatalogResult:
        return [], {"source": "static"}

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        self._validate_provider(request)
        self._validate_mode(request)
        self._validate_max_tokens(request)
        return self.controls(request.model).validate_request(request)

    def generate(self, request: LlmRequest) -> LlmResponse:
        warnings = self.validate_request(request)
        response = self._provider.generate(request)
        if not warnings:
            return response
        return response.model_copy(
            update={"warnings": [*response.warnings, *warnings]}
        )

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

    def _validate_max_tokens(self, request: LlmRequest) -> None:
        if request.max_tokens is not None and self.mode == CallMode.headless:
            raise ValueError(
                f"max_tokens is not supported for provider={self.name!r}"
            )

    def _validate_max_tokens_required(self, request: LlmRequest) -> None:
        max_tokens = request.max_tokens
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
