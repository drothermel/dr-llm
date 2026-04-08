from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dr_llm.logging.config import GenerationLogConfig
from dr_llm.logging.events import GenerationLogEvent, get_generation_log_context
from dr_llm.logging.redaction import redact_payload


LOG_FILE_NAME = "generation_transcripts.jsonl"
TRUNCATION_KEY = "_dr_llm_truncated"
_ERROR_SUPPRESSION_WINDOW = timedelta(seconds=30)
_LOGGER = logging.getLogger(__name__)


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _truncation_envelope(payload: dict[str, Any], raw: bytes, max_bytes: int) -> str:
    # Reserve half of max_bytes for the inlined event_prefix so the envelope's
    # constant header fields plus the prefix together still fit inside
    # max_bytes. Without this floor (see config.MIN_MAX_EVENT_BYTES) the
    # truncation envelope itself could overflow the configured cap.
    return json.dumps(
        {
            TRUNCATION_KEY: True,
            "max_event_bytes": max_bytes,
            "event_type": payload.get("event_type"),
            "stage": payload.get("stage"),
            "event_prefix": raw[: max_bytes // 2].decode("utf-8", errors="ignore"),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def encode_or_truncate(payload: dict[str, Any], max_bytes: int) -> str:
    encoded = _serialize(payload)
    raw = encoded.encode("utf-8")
    if len(raw) <= max_bytes:
        return encoded
    return _truncation_envelope(payload, raw, max_bytes)


class GenerationLogSink:
    def __init__(self, config: GenerationLogConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._last_error_emit_utc: datetime | None = None

    @property
    def log_path(self) -> Path:
        return self.config.log_dir / LOG_FILE_NAME

    def emit(self, event: GenerationLogEvent) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            record = self._build_record(event)
            line = encode_or_truncate(record, self.config.max_event_bytes)
            try:
                self._write_line(line)
            except Exception as exc:  # noqa: BLE001
                self._emit_sink_error(exc)

    def emit_event(
        self, *, event_type: str, stage: str, payload: dict[str, Any]
    ) -> None:
        event = GenerationLogEvent.from_context(
            event_type=event_type,
            stage=stage,
            payload=payload,
            context=get_generation_log_context(),
        )
        self.emit(event)

    def _build_record(self, event: GenerationLogEvent) -> dict[str, Any]:
        record = event.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        record["payload"] = redact_payload(
            record.get("payload", {}),
            enabled=self.config.redact_enabled,
        )
        return record

    def _write_line(self, line: str) -> None:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.write("\n")

    def _rotate_if_needed(self) -> None:
        if self.config.rotate_bytes <= 0 or not self.log_path.exists():
            return
        if self.log_path.stat().st_size < self.config.rotate_bytes:
            return
        if self.config.backups <= 0:
            self._drop_log()
            return
        self._shift_backups()

    def _drop_log(self) -> None:
        self.log_path.unlink(missing_ok=True)

    def _shift_backups(self) -> None:
        for idx in range(self.config.backups - 1, 0, -1):
            src = self.config.log_dir / f"{LOG_FILE_NAME}.{idx}"
            dst = self.config.log_dir / f"{LOG_FILE_NAME}.{idx + 1}"
            if src.exists():
                src.replace(dst)
        first_backup = self.config.log_dir / f"{LOG_FILE_NAME}.1"
        self.log_path.replace(first_backup)

    def _emit_sink_error(self, exc: Exception) -> None:
        now = datetime.now(timezone.utc)
        if (
            self._last_error_emit_utc is not None
            and now - self._last_error_emit_utc < _ERROR_SUPPRESSION_WINDOW
        ):
            return
        self._last_error_emit_utc = now
        _LOGGER.error("generation log sink error: %s", exc)


@lru_cache(maxsize=1)
def get_generation_log_sink() -> GenerationLogSink:
    """Return the process-wide generation log sink.

    The sink is built once per process from ``GenerationLogConfig()`` (which
    reads environment variables at construction time). Later env changes do
    not apply until :func:`reset_generation_log_sink` clears the cache.
    """
    return GenerationLogSink(GenerationLogConfig())


def reset_generation_log_sink() -> None:
    """Clear the cached sink so the next access rebuilds from current config/env.

    ``GenerationLogConfig`` is resolved when the sink is first constructed; call
    this after changing environment variables (for example in tests) so
    :func:`get_generation_log_sink` picks up new settings.
    """
    get_generation_log_sink.cache_clear()


def emit_generation_event(
    *, event_type: str, stage: str, payload: dict[str, Any]
) -> None:
    sink = get_generation_log_sink()
    sink.emit_event(event_type=event_type, stage=stage, payload=payload)
