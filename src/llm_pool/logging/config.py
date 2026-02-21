from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class GenerationLogConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    log_dir: Path = Path(".llm_pool/generation_logs")
    rotate_bytes: int = 100 * 1024 * 1024
    backups: int = 10
    redact_secrets: bool = True
    max_event_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> GenerationLogConfig:
        return cls(
            enabled=_env_bool("LLM_POOL_GENERATION_LOG_ENABLED", True),
            log_dir=Path(
                os.getenv(
                    "LLM_POOL_GENERATION_LOG_DIR",
                    ".llm_pool/generation_logs",
                )
            ),
            rotate_bytes=_env_int(
                "LLM_POOL_GENERATION_LOG_ROTATE_BYTES",
                100 * 1024 * 1024,
            ),
            backups=_env_int("LLM_POOL_GENERATION_LOG_BACKUPS", 10),
            redact_secrets=_env_bool("LLM_POOL_GENERATION_LOG_REDACT_SECRETS", True),
            max_event_bytes=_env_int(
                "LLM_POOL_GENERATION_LOG_MAX_EVENT_BYTES",
                10 * 1024 * 1024,
            ),
        )
