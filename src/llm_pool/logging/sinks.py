from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_pool.logging.config import GenerationLogConfig
from llm_pool.logging.events import GenerationLogEvent, get_generation_log_context
from llm_pool.logging.redaction import redact_payload


LOG_FILE_NAME = "generation_transcripts.jsonl"
TRUNCATION_KEY = "_llm_pool_truncated"


class GenerationLogSink:
    def __init__(self, config: GenerationLogConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._last_error_emit_utc: datetime | None = None
        self._error_suppression_window = timedelta(seconds=30)

    @property
    def log_path(self) -> Path:
        return self.config.log_dir / LOG_FILE_NAME

    def emit(self, event: GenerationLogEvent) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            try:
                self.config.log_dir.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                payload = event.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_computed_fields=True,
                )
                payload["payload"] = redact_payload(
                    payload.get("payload", {}),
                    enabled=self.config.redact_secrets,
                )
                line = self._encode_payload(payload)
                with self.log_path.open("a", encoding="utf-8") as file:
                    file.write(line)
                    file.write("\n")
            except Exception as exc:  # noqa: BLE001
                self._emit_sink_error(exc)

    def emit_event(
        self, *, event_type: str, stage: str, payload: dict[str, Any]
    ) -> None:
        context = get_generation_log_context()
        event = GenerationLogEvent(
            event_type=event_type,
            stage=stage,
            call_id=context.get("call_id"),
            run_id=context.get("run_id"),
            provider=context.get("provider"),
            model=context.get("model"),
            mode=context.get("mode"),
            thread_id=context.get("thread_id"),
            payload=payload,
        )
        self.emit(event)

    def _encode_payload(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        max_bytes = max(self.config.max_event_bytes, 1024)
        raw = encoded.encode("utf-8")
        if len(raw) <= max_bytes:
            return encoded
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

    def _rotate_if_needed(self) -> None:
        if self.config.rotate_bytes <= 0 or not self.log_path.exists():
            return
        if self.log_path.stat().st_size < self.config.rotate_bytes:
            return
        backups = max(0, self.config.backups)
        if backups == 0:
            self.log_path.unlink(missing_ok=True)
            return
        for idx in range(backups - 1, 0, -1):
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
            and now - self._last_error_emit_utc < self._error_suppression_window
        ):
            return
        self._last_error_emit_utc = now
        print(
            f"[llm-pool] generation log sink error: {exc}",
            file=sys.stderr,
        )


_SINK_LOCK = threading.RLock()
_GLOBAL_SINK: GenerationLogSink | None = None


def get_generation_log_sink() -> GenerationLogSink:
    global _GLOBAL_SINK
    with _SINK_LOCK:
        if _GLOBAL_SINK is None:
            _GLOBAL_SINK = GenerationLogSink(GenerationLogConfig.from_env())
        return _GLOBAL_SINK


def emit_generation_event(
    *, event_type: str, stage: str, payload: dict[str, Any]
) -> None:
    sink = get_generation_log_sink()
    sink.emit_event(event_type=event_type, stage=stage, payload=payload)
