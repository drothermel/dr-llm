from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from collections.abc import Generator
from threading import get_ident
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# Single source of truth for which context fields propagate from the active
# generation_log_context() into a logged event. Adding a new propagated field
# means appending it here and adding the matching attribute below — nothing
# else in the logging stack needs to change. thread_id is intentionally absent:
# it is captured at emit time so it reflects the emitting thread, not the
# thread that opened the context (ContextVars propagate across awaits and
# threads).
_CONTEXT_FIELDS: tuple[str, ...] = (
    "call_id",
    "run_id",
    "provider",
    "model",
    "mode",
)


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
    thread_id: int = Field(default_factory=get_ident)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_context(
        cls,
        *,
        event_type: str,
        stage: str,
        payload: dict[str, Any],
        context: Mapping[str, Any],
    ) -> "GenerationLogEvent":
        data: dict[str, Any] = {
            "event_type": event_type,
            "stage": stage,
            "payload": payload,
        }
        for field in _CONTEXT_FIELDS:
            data[field] = context.get(field)
        return cls.model_validate(data)


_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})

_GENERATION_LOG_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "dr_llm_generation_log_context",
    default=_EMPTY_CONTEXT,
)


def get_generation_log_context() -> dict[str, Any]:
    return dict(_GENERATION_LOG_CONTEXT.get())


@contextmanager
def generation_log_context(values: dict[str, Any]) -> Generator[None, None, None]:
    merged = {**_GENERATION_LOG_CONTEXT.get(), **values}
    token = _GENERATION_LOG_CONTEXT.set(MappingProxyType(merged))
    try:
        yield
    finally:
        _GENERATION_LOG_CONTEXT.reset(token)
