from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from dr_llm.cli import app
import dr_llm.streaming_log.cli as streaming_log_cli
from dr_llm.streaming_log.bootstrap import StreamingLogStatus
from dr_llm.streaming_log.ingest_pools import PoolImportResult
from dr_llm.streaming_log.workers import StreamingWorkerConfig

runner = CliRunner()


def _status() -> StreamingLogStatus:
    return StreamingLogStatus(
        nats_url="nats://example:4222",
        events_stream="DRLLM_EVENTS",
        work_stream="DRLLM_WORK",
        payload_bucket="DRLLM_PAYLOADS",
        events_subjects=["drllm.events.>"],
        work_subjects=["drllm.work.>"],
        payload_objects=3,
    )


def test_streaming_log_bootstrap_emits_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_bootstrap_streaming_log() -> StreamingLogStatus:
        nonlocal calls
        calls += 1
        return _status()

    monkeypatch.setattr(
        streaming_log_cli,
        "bootstrap_streaming_log",
        fake_bootstrap_streaming_log,
    )

    result = runner.invoke(app, ["streaming-log", "bootstrap"])

    assert result.exit_code == 0
    assert calls == 1
    assert json.loads(result.stdout)["payload_objects"] == 3


def test_streaming_log_inspect_emits_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_inspect_streaming_log() -> StreamingLogStatus:
        nonlocal calls
        calls += 1
        return _status()

    monkeypatch.setattr(
        streaming_log_cli,
        "inspect_streaming_log",
        fake_inspect_streaming_log,
    )

    result = runner.invoke(app, ["streaming-log", "inspect"])

    assert result.exit_code == 0
    assert calls == 1
    assert json.loads(result.stdout)["events_stream"] == "DRLLM_EVENTS"


def test_streaming_log_ingest_pool_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingested: list[tuple[str, str, str | None, int | None]] = []

    async def fake_ingest_one_pool(
        *,
        dsn: str,
        pool_name: str,
        source_id: str | None,
        sample_limit: int | None,
    ) -> PoolImportResult:
        ingested.append((dsn, pool_name, source_id, sample_limit))
        return PoolImportResult(
            pool_name=pool_name,
            imported_count=2,
            event_ids=["event-1", "event-2"],
        )

    monkeypatch.setattr(
        streaming_log_cli, "_ingest_one_pool", fake_ingest_one_pool
    )

    result = runner.invoke(
        app,
        [
            "streaming-log",
            "ingest-pool",
            "--dsn",
            "postgresql://example/demo",
            "--pool-name",
            "pool-1",
            "--source-id",
            "source-1",
            "--sample-limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert ingested == [("postgresql://example/demo", "pool-1", "source-1", 2)]
    assert json.loads(result.stdout)["imported_count"] == 2


def test_streaming_log_ingest_pools_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingested: list[tuple[str, str | None, int | None]] = []

    async def fake_ingest_all_pools(
        *, dsn: str, source_id: str | None, sample_limit: int | None
    ) -> list[PoolImportResult]:
        ingested.append((dsn, source_id, sample_limit))
        return [
            PoolImportResult(pool_name="pool-1", imported_count=1),
            PoolImportResult(pool_name="pool-2", imported_count=3),
        ]

    monkeypatch.setattr(
        streaming_log_cli, "_ingest_all_pools", fake_ingest_all_pools
    )

    result = runner.invoke(
        app,
        [
            "streaming-log",
            "ingest-pools",
            "--dsn",
            "postgresql://example/demo",
            "--source-id",
            "source-1",
            "--sample-limit",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert ingested == [("postgresql://example/demo", "source-1", 5)]
    payload = json.loads(result.stdout)
    assert [pool["pool_name"] for pool in payload["pools"]] == [
        "pool-1",
        "pool-2",
    ]


def test_streaming_log_run_worker_forwards_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs: list[StreamingWorkerConfig] = []

    async def fake_run_streaming_worker(
        *, config: StreamingWorkerConfig
    ) -> None:
        configs.append(config)

    monkeypatch.setattr(
        streaming_log_cli,
        "run_streaming_worker",
        fake_run_streaming_worker,
    )

    result = runner.invoke(
        app,
        [
            "streaming-log",
            "run-worker",
            "--worker-id",
            "worker-1",
            "--max-messages",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert configs == [
        StreamingWorkerConfig(worker_id="worker-1", max_messages=2)
    ]
