import json
from datetime import datetime, timezone

from typer.testing import CliRunner

import llm_pool.cli as cli_module
from llm_pool.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    KnownOperationName,
    OperationMix,
    OperationStats,
    PhaseStats,
)
from llm_pool.cli import app
from llm_pool.types import RunStatus


def test_providers_command_lists_known_providers() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload["providers"]}
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "glm" in providers
    assert "minimax" in providers
    assert "claude-code-minimax" in providers
    assert "claude-code-kimi" in providers


class _CliFakeRepository:
    def initialize(self) -> None:
        return None

    def start_run(self, **_: object) -> str:
        return "run_cli"

    def upsert_run_parameters(
        self, *, run_id: str, parameters: dict[str, object]
    ) -> int:
        _ = run_id, parameters
        return 1

    def record_call(self, **_: object) -> str:
        return "call_cli"

    def finish_run(self, **_: object) -> None:
        return None

    def list_calls(self, **_: object) -> list[object]:
        return []

    def record_artifact(self, **_: object) -> str:
        return "artifact_cli"

    def start_session(self, **_: object) -> object:
        class _Handle:
            session_id = "session_cli"

        return _Handle()

    def create_session_turn(self, **_: object) -> tuple[str, int]:
        return ("turn_cli", 1)

    def append_session_event(self, **_: object) -> str:
        return "event_cli"

    def complete_session_turn(self, **_: object) -> None:
        return None

    def update_session_status(self, **_: object) -> None:
        return None

    def close(self) -> None:
        return None


def test_run_benchmark_outputs_summary(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    fake_repo = _CliFakeRepository()
    captured: dict[str, BenchmarkConfig] = {}
    artifact_path = str(tmp_path / "report.json")

    def fake_repo_builder(
        dsn: str | None, min_pool_size: int, max_pool_size: int
    ) -> _CliFakeRepository:
        _ = dsn, min_pool_size, max_pool_size
        return fake_repo

    def fake_run_benchmark(
        *, repository: object, config: BenchmarkConfig
    ) -> BenchmarkReport:
        _ = repository
        captured["config"] = config

        by_op: dict[KnownOperationName, OperationStats] = {
            "record_call": OperationStats(executed=1350, failed=0),
            "session_roundtrip": OperationStats(executed=675, failed=0),
            "read_calls": OperationStats(executed=675, failed=0),
        }
        measured = PhaseStats(
            phase="measured",
            total_operations=2700,
            successful_operations=2700,
            failed_operations=0,
            elapsed_ms=21951,
            operations_per_second=123.0,
            p50_latency_ms=10.0,
            p95_latency_ms=50.0,
            by_operation=by_op,
        )
        warmup_by_op: dict[KnownOperationName, OperationStats] = {
            "record_call": OperationStats(executed=100, failed=0),
            "session_roundtrip": OperationStats(executed=50, failed=0),
            "read_calls": OperationStats(executed=50, failed=0),
        }
        warmup = PhaseStats(
            phase="warmup",
            total_operations=300,
            successful_operations=300,
            failed_operations=0,
            elapsed_ms=1000,
            operations_per_second=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            by_operation=warmup_by_op,
        )
        now = datetime.now(timezone.utc)
        return BenchmarkReport(
            run_id="run_cli",
            status=RunStatus.success,
            started_at=now,
            finished_at=now,
            config=config,
            warmup=warmup,
            measured=measured,
            errors_sampled=[],
            artifact_path=artifact_path,
        )

    monkeypatch.setattr(cli_module, "_repo", fake_repo_builder)
    monkeypatch.setattr(cli_module, "run_repository_benchmark", fake_run_benchmark)

    result = runner.invoke(
        app,
        [
            "run",
            "benchmark",
            "--workers",
            "12",
            "--total-operations",
            "3000",
            "--warmup-operations",
            "300",
            "--max-in-flight",
            "12",
            "--operation-mix-json",
            '{"record_call":2,"session_roundtrip":1,"read_calls":1}',
            "--artifact-path",
            artifact_path,
            "--max-failure-ratio",
            "0.25",
            "--max-error-samples",
            "17",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "run_cli"
    assert payload["status"] == "success"
    assert payload["operations_per_second"] == 123.0
    assert payload["failed_operations"] == 0
    assert payload["artifact_path"] == artifact_path
    config = captured["config"]
    assert config.total_operations == 3000
    assert config.warmup_operations == 300
    assert config.max_in_flight == 12
    assert config.max_failure_ratio == 0.25
    assert config.max_error_samples == 17
    assert config.operation_mix == OperationMix(
        record_call=2, session_roundtrip=1, read_calls=1
    )


def test_run_benchmark_rejects_invalid_operation_mix(
    monkeypatch,
) -> None:
    runner = CliRunner()

    monkeypatch.setattr(cli_module, "_repo", lambda *_: _CliFakeRepository())

    result = runner.invoke(
        app,
        [
            "run",
            "benchmark",
            "--operation-mix-json",
            '{"record_call":0,"session_roundtrip":0,"read_calls":0}',
        ],
    )

    assert result.exit_code == 2
