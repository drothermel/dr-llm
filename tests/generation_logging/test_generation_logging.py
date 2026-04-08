from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dr_llm.logging.config import GenerationLogConfig
from dr_llm.logging.events import GenerationLogEvent
from dr_llm.logging.sinks import GenerationLogSink


def test_generation_log_sink_writes_redacted_jsonl(tmp_path: Path) -> None:
    sink = GenerationLogSink(
        GenerationLogConfig(
            enabled=True,
            log_dir=tmp_path,
            rotate_bytes=1024 * 1024,
            backups=2,
            redact_enabled=True,
            max_event_bytes=1024 * 1024,
        )
    )
    sink.emit(
        GenerationLogEvent(
            event_type="provider.raw_response",
            stage="openai.http_response",
            call_id="c1",
            payload={
                "headers": {"Authorization": "Bearer super-secret"},
                "request": {"api_key": "secret"},
                "response_text": "ok",
            },
        )
    )
    lines = (
        (tmp_path / "generation_transcripts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["payload"]["headers"]["Authorization"] == "[REDACTED]"
    assert payload["payload"]["request"]["api_key"] == "[REDACTED]"


def test_generation_log_sink_is_thread_safe(tmp_path: Path) -> None:
    sink = GenerationLogSink(
        GenerationLogConfig(
            enabled=True,
            log_dir=tmp_path,
            rotate_bytes=1024 * 1024 * 1024,
            backups=2,
            redact_enabled=False,
            max_event_bytes=1024 * 1024,
        )
    )
    total = 200

    def emit_one(index: int) -> None:
        sink.emit(
            GenerationLogEvent(
                event_type="llm_call.succeeded",
                stage="client.after_adapter",
                call_id=f"c{index}",
                payload={"i": index},
            )
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = []
        for idx in range(total):
            futures.append(pool.submit(emit_one, idx))
        for future in futures:
            future.result()

    lines = (
        (tmp_path / "generation_transcripts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == total
    seen = set()
    for line in lines:
        payload = json.loads(line)
        assert payload["event_type"] == "llm_call.succeeded"
        seen.add(payload["call_id"])
    assert len(seen) == total
