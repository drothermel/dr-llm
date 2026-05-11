from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

from dr_llm.llm.names import EffortSpec, ProviderName
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.response import CallMode

if TYPE_CHECKING:
    from dr_llm.llm.request import Message
    from dr_llm.llm.providers.core.protocol import ProviderOrchestrator


class SamplingControls(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = None
    top_p: float | None = None

    def is_empty(self) -> bool:
        return self.temperature is None and self.top_p is None


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    model: str
    mode: CallMode
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None


LLM_CONFIG_ADAPTER = TypeAdapter(LlmConfig)


def parse_llm_config(payload: object) -> LlmConfig:
    return LLM_CONFIG_ADAPTER.validate_python(payload)


def build_request_from_config(
    orchestrator: ProviderOrchestrator,
    config: LlmConfig,
    messages: list[Message],
    *,
    metadata: dict[str, Any] | None = None,
):
    return orchestrator.build_request_from_config(
        config=config,
        messages=messages,
        metadata=metadata,
    )
