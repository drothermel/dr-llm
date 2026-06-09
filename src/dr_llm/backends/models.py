"""Pydantic models for the backends public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import (
    CallMode,
    EffortSpec,
    Message,
    ProviderName,
    ReasoningSpec,
    SamplingControls,
    TokenUsage,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.providers.core.usage import CostInfo
from dr_llm.workers.models import WorkerStatCounts

ResponseSource = Literal["direct", "pool_cache", "generated"]


class BackendRequest(BaseModel):
    """Caller-facing LLM request with optional v1 extensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    model: str
    mode: CallMode
    messages: list[Message]
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)

    def to_llm_request(self) -> LlmRequest:
        return LlmRequest(
            provider=self.provider,
            model=self.model,
            mode=self.mode,
            messages=self.messages,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
            sampling=self.sampling,
            metadata=self.metadata,
        )

    @classmethod
    def from_llm_request(
        cls,
        llm_request: LlmRequest,
        *,
        extensions: dict[str, Any] | None = None,
    ) -> BackendRequest:
        return cls(
            provider=llm_request.provider,
            model=llm_request.model,
            mode=llm_request.mode,
            messages=llm_request.messages,
            max_tokens=llm_request.max_tokens,
            effort=llm_request.effort,
            reasoning=llm_request.reasoning,
            sampling=llm_request.sampling,
            metadata=llm_request.metadata,
            extensions=extensions or {},
        )


class BackendResponse(BaseModel):
    """LLM response with backend provenance metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    cost: CostInfo | None = None
    raw_json: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    provider: str
    model: str
    mode: CallMode
    warnings: list[ReasoningWarning] = Field(default_factory=list)
    source: ResponseSource | None = None
    sample_id: str | None = None
    request_fingerprint: str | None = None


class BackendCapabilities(BaseModel):
    """Snapshot of provider controls for a model."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    mode: CallMode
    control_mode: str
    supported_thinking_levels: tuple[str, ...] = ()
    default_thinking_level: str | None = None
    supported_effort_levels: tuple[str, ...] = ()
    default_effort: str | None = None
    default_reasoning: dict[str, Any] | None = None
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    catalog_metadata: dict[str, Any] = Field(default_factory=dict)


class AcquireResult(BaseModel):
    """Result of a pool session acquire."""

    model_config = ConfigDict(frozen=True)

    responses: list[BackendResponse] = Field(default_factory=list)
    claimed_from_cache: int = 0
    generated: int = 0


class SubmitResult(BaseModel):
    """Result of seeding incomplete pool rows for batch fill."""

    model_config = ConfigDict(frozen=True)

    seeded: int = 0
    skipped: int = 0


class DrainResult(BaseModel):
    """Result of draining incomplete pool samples via workers."""

    model_config = ConfigDict(frozen=True)

    incomplete: int = 0
    complete: int = 0
    worker_counts: WorkerStatCounts = Field(default_factory=WorkerStatCounts)


class PoolBackendConfig(BaseModel):
    """Configuration for :class:`PoolBackend`."""

    model_config = ConfigDict(frozen=True)

    pool_name: str
    database_url: str | None = None
    consumer_id: str | None = None
    num_workers: int = Field(default=4, gt=0)
    lease_seconds: int = Field(default=300, gt=0)
    acquire_timeout_seconds: float = Field(default=60.0, gt=0)
