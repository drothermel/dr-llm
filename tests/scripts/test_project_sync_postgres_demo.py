from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from dr_llm.demo.projects import DemoDsnLease


runner = CliRunner()


def _load_sync_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-project-sync-postgres.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_project_sync_postgres", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_command_syncs_and_verifies_target_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_demo = _load_sync_demo()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        sync_demo,
        "uuid4",
        lambda: type("Uuid", (), {"hex": "abcdef123456"})(),
    )

    def fake_prepare_demo_dsn(**kwargs: object) -> DemoDsnLease:
        calls.append(("prepare", kwargs))
        return DemoDsnLease(
            dsn="postgresql://user:pass@example/source?sslmode=require",
            project_name=str(kwargs["project_name"]),
            should_destroy_project=True,
        )

    def fake_run_dr_llm_streaming(*args: str) -> None:
        calls.append(("sync", args))

    def fake_verify_target_pool(dsn: str, expected_sample_id: str) -> None:
        calls.append(("verify", (dsn, expected_sample_id)))

    monkeypatch.setattr(sync_demo, "prepare_demo_dsn", fake_prepare_demo_dsn)
    monkeypatch.setattr(
        sync_demo, "_seed_source_pool", lambda _dsn: "sample-1"
    )
    monkeypatch.setattr(
        sync_demo, "run_dr_llm_streaming", fake_run_dr_llm_streaming
    )
    monkeypatch.setattr(
        sync_demo, "_verify_target_pool", fake_verify_target_pool
    )
    monkeypatch.setattr(
        sync_demo,
        "cleanup_demo_dsn",
        lambda lease: calls.append(("cleanup", lease.project_name)),
    )

    result = runner.invoke(sync_demo.app)

    assert result.exit_code == 0
    assert any(name == "sync" for name, _ in calls)

    verify_args = next(args for name, args in calls if name == "verify")
    assert isinstance(verify_args, tuple)
    _, expected_sample_id = verify_args
    assert expected_sample_id == "sample-1"
    assert any(name == "cleanup" for name, _ in calls)
