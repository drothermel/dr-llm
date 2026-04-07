from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

TBackendState = TypeVar("TBackendState", bound=BaseModel)


type WorkerStatKey = Literal[
    "claimed",
    "completed",
    "failed",
    "retried",
    "process_errors",
    "idle_polls",
]

WORKER_STAT_KEYS: tuple[WorkerStatKey, ...] = (
    "claimed",
    "completed",
    "failed",
    "retried",
    "process_errors",
    "idle_polls",
)


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    num_workers: int = Field(gt=0)
    lease_seconds: int = Field(default=300, gt=0)
    min_poll_interval_s: float = Field(default=0.5, gt=0)
    max_poll_interval_s: float = Field(default=5.0, gt=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    thread_name_prefix: str = "worker"

    @model_validator(mode="after")
    def _validate(self) -> WorkerConfig:
        if self.max_poll_interval_s < self.min_poll_interval_s:
            raise ValueError("max_poll_interval_s must be >= min_poll_interval_s")
        if not self.thread_name_prefix.strip():
            raise ValueError("thread_name_prefix must be non-empty")
        return self


class WorkerStatCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    claimed: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    process_errors: int = 0
    idle_polls: int = 0


class WorkerSnapshot(BaseModel, Generic[TBackendState]):
    """Observable state for a running worker controller."""

    model_config = ConfigDict(frozen=True)

    worker_count: int
    stop_requested: bool = False
    counts: WorkerStatCounts = Field(default_factory=WorkerStatCounts)
    backend_state: TBackendState | None = None
