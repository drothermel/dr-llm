from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


type WorkerStatKey = Literal[
    "claimed",
    "completed",
    "failed",
    "retried",
    "process_errors",
    "idle_polls",
]


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    num_workers: int
    lease_seconds: int = 300
    min_poll_interval_s: float = 0.5
    max_poll_interval_s: float = 5.0
    backoff_factor: float = 2.0
    max_retries: int = 0
    thread_name_prefix: str = "worker"

    @model_validator(mode="after")
    def _validate(self) -> WorkerConfig:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.min_poll_interval_s <= 0 or self.max_poll_interval_s <= 0:
            raise ValueError("poll intervals must be positive")
        if self.max_poll_interval_s < self.min_poll_interval_s:
            raise ValueError("max_poll_interval_s must be >= min_poll_interval_s")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not self.thread_name_prefix.strip():
            raise ValueError("thread_name_prefix must be non-empty")
        return self


class WorkerStatCounts(BaseModel):
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    process_errors: int = 0
    idle_polls: int = 0

    def increment(self, key: WorkerStatKey, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)


class WorkerSnapshot(BaseModel):
    """Observable state for a running worker controller."""

    model_config = ConfigDict(frozen=True)

    worker_count: int
    stop_requested: bool = False
    counts: WorkerStatCounts = Field(default_factory=WorkerStatCounts)
    backend_state: BaseModel | None = None
