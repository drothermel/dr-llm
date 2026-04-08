from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Floor matching the half-prefix reservation in sinks.encode_or_truncate;
# the rationale lives next to the truncation logic.
MIN_MAX_EVENT_BYTES = 1024


class GenerationLogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DR_LLM_GENERATION_LOG_",
        frozen=True,
        populate_by_name=True,
    )

    enabled: bool = True
    log_dir: Path = Field(
        default=Path(".dr_llm/generation_logs"),
        validation_alias="DR_LLM_GENERATION_LOG_DIR",
    )
    rotate_bytes: int = 100 * 1024 * 1024
    backups: int = 10
    redact_enabled: bool = True
    max_event_bytes: int = Field(default=10 * 1024 * 1024, ge=MIN_MAX_EVENT_BYTES)
