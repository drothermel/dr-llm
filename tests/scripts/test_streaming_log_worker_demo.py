from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from typer.testing import CliRunner

from dr_llm.llm import ProviderName


runner = CliRunner()


def _load_worker_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_worker", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_demo_command_forwards_provider_options(monkeypatch) -> None:
    worker_demo = _load_worker_demo()
    calls: list[object] = []

    async def fake_run_worker_demo(options: object) -> None:
        calls.append(options)

    monkeypatch.setattr(
        worker_demo,
        "_run_worker_demo",
        fake_run_worker_demo,
    )

    result = runner.invoke(
        worker_demo.app,
        [
            "--nats-url",
            "nats://localhost:4222",
            "--keep-nats",
            "--prompt",
            "hello",
            "--max-retries",
            "2",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    options = calls[0]
    assert isinstance(options, worker_demo.WorkerDemoOptions)
    assert options.nats.nats_url == "nats://localhost:4222"
    assert options.nats.keep_nats
    assert options.provider == ProviderName.ANTHROPIC
    assert options.prompt == "hello"
    assert options.max_retries == 2
    assert options.model == "claude-test"
