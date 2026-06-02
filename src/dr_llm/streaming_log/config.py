from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StreamingLogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DR_LLM_STREAMING_LOG_",
        extra="ignore",
        frozen=True,
    )

    nats_url: str = "nats://localhost:4222"
    events_stream: str = "DRLLM_EVENTS"
    work_stream: str = "DRLLM_WORK"
    payload_bucket: str = "DRLLM_PAYLOADS"
    events_subject: str = "drllm.events.>"
    work_subject: str = "drllm.work.>"
    llm_work_subject: str = "drllm.work.llm"
    work_consumer: str = "drllm_work_workers"
    event_consumer: str = "drllm_events_replay"
    ack_wait_seconds: float = Field(default=300.0, gt=0)
    max_deliver: int = Field(default=3, ge=1)
    fetch_batch_size: int = Field(default=10, ge=1)
    inline_payload_threshold: int = Field(default=65536, ge=0)
    producer_name: str = "dr-llm"
    producer_version: str | None = None


class NatsResourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    events_stream: str
    work_stream: str
    payload_bucket: str


__all__ = ["NatsResourceStatus", "StreamingLogConfig"]
