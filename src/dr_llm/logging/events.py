from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import get_ident
from typing import Any, Generator

from pydantic import BaseModel, ConfigDict, Field


class GenerationLogEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: str
    stage: str
    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    call_id: str | None = None
    run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    mode: str | None = None
    thread_id: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


_GENERATION_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "dr_llm_generation_log_context",
    default=None,
)


def get_generation_log_context() -> dict[str, Any]:
    current = _GENERATION_LOG_CONTEXT.get()
    return dict(current) if isinstance(current, dict) else {}


@contextmanager
def generation_log_context(values: dict[str, Any]) -> Generator[None, None, None]:
    current = get_generation_log_context()
    merged = {**current, **values}
    merged.setdefault("thread_id", get_ident())
    token = _GENERATION_LOG_CONTEXT.set(dict(merged))
    try:
        yield
    finally:
        _GENERATION_LOG_CONTEXT.reset(token)
