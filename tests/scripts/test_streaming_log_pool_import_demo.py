from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from typer.testing import CliRunner


runner = CliRunner()


def _load_pool_import_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-pool-import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_pool_import", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pool_import_demo_command_forwards_options(monkeypatch) -> None:
    pool_import_demo = _load_pool_import_demo()
    calls: list[str] = []

    async def fake_run_import_demo(options: Any) -> None:
        assert options.dsn == "postgresql://localhost/demo"
        assert options.pool_name == "demo_pool"
        assert options.nats.nats_url == "nats://localhost:4222"
        assert options.nats.keep_nats
        assert options.source_id == "source"
        assert options.sample_limit == 3
        assert options.event_sample_limit == 2
        calls.append("run")

    monkeypatch.setattr(
        pool_import_demo,
        "_run_import_demo",
        fake_run_import_demo,
    )

    result = runner.invoke(
        pool_import_demo.app,
        [
            "--dsn",
            "postgresql://localhost/demo",
            "--pool-name",
            "demo_pool",
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--source-id",
            "source",
            "--sample-limit",
            "3",
            "--event-sample-limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["run"]
