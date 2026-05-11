from __future__ import annotations

from typing import Any, Protocol

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import EffortSpec, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class ProviderControls(Protocol):
    provider: Any
    model: str
    mode: CallMode

    @property
    def supports_reasoning(self) -> bool: ...

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]: ...

    @property
    def default_thinking_level(self) -> ThinkingLevel: ...

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]: ...

    @property
    def default_effort(self) -> EffortSpec: ...

    @property
    def default_reasoning(self) -> ReasoningSpec | None: ...

    @property
    def catalog_metadata(self) -> dict[str, Any]: ...

    def request_defaults(self) -> ProviderRequestDefaults: ...

    def resolve_reasoning(
        self,
        *,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None: ...

    def resolve_effort(self, effort: EffortSpec | None) -> EffortSpec: ...

    def resolve_sampling(
        self, sampling: SamplingControls | None
    ) -> SamplingControls | None: ...

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None: ...

    def validate_request(
        self, request: LlmRequest
    ) -> list[ReasoningWarning]: ...
