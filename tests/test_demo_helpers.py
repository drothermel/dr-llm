from __future__ import annotations

import subprocess
from typing import Any

import pytest
import typer
from pydantic import ValidationError

import dr_llm.demo.cli_calls as demo_cli_calls
import dr_llm.demo.counts as demo_counts
import dr_llm.demo.projects as demo_projects
import dr_llm.demo.requirements as demo_requirements
from dr_llm.pool import LlmPoolBackendState
from dr_llm.project import CreateProjectRequest, ProjectInfo
from dr_llm.project.errors import DockerUnavailableError
from dr_llm.workers import WorkerSnapshot, WorkerStatCounts


def test_demo_counts_formats_default_attempt_summary() -> None:
    counts = demo_counts.DemoCounts()

    assert counts.format_line(demo_counts.ATTEMPT_SUMMARY_FIELDS) == (
        "attempted=0 succeeded=0 failed=0 had_output_text=0"
    )


def test_demo_counts_increment_updates_selected_metric() -> None:
    counts = demo_counts.DemoCounts()

    counts.increment("attempted")
    counts.increment("failed", by=2)

    assert counts.attempted == 1
    assert counts.failed == 2


def test_demo_counts_increment_rejects_negative_result() -> None:
    counts = demo_counts.DemoCounts()

    with pytest.raises(ValidationError):
        counts.increment("attempted", by=-1)


def test_demo_counts_formats_unknown_pool_totals() -> None:
    counts = demo_counts.DemoCounts(claimed=1, completed=2, failed=3)

    assert counts.format_line(demo_counts.POOL_PROGRESS_FIELDS) == (
        "claimed=1 completed=2 failed=3 incomplete=? complete=?"
    )
    assert counts.key(demo_counts.POOL_PROGRESS_FIELDS) == (
        1,
        2,
        3,
        None,
        None,
    )


def test_demo_counts_changed_from_compares_selected_fields() -> None:
    previous = demo_counts.DemoCounts(attempted=1, claimed=7)
    unchanged = demo_counts.DemoCounts(attempted=1, claimed=8)
    changed = demo_counts.DemoCounts(attempted=2, claimed=7)

    fields = demo_counts.ATTEMPT_SUMMARY_FIELDS

    assert previous.changed_from(None, fields)
    assert not unchanged.changed_from(previous, fields)
    assert changed.changed_from(previous, fields)


def test_demo_counts_from_pool_snapshot_with_backend_state() -> None:
    snapshot = WorkerSnapshot[LlmPoolBackendState](
        worker_count=2,
        counts=WorkerStatCounts(claimed=3, completed=2, failed=1),
        backend_state=LlmPoolBackendState(incomplete=4, complete=5),
    )

    counts = demo_counts.DemoCounts.from_pool_snapshot(snapshot)

    assert counts.format_line(demo_counts.POOL_PROGRESS_FIELDS) == (
        "claimed=3 completed=2 failed=1 incomplete=4 complete=5"
    )


def test_demo_counts_from_pool_snapshot_without_backend_state() -> None:
    snapshot = WorkerSnapshot[LlmPoolBackendState](
        worker_count=2,
        counts=WorkerStatCounts(claimed=3, completed=2, failed=1),
    )

    counts = demo_counts.DemoCounts.from_pool_snapshot(snapshot)

    assert counts.format_line(demo_counts.POOL_PROGRESS_FIELDS) == (
        "claimed=3 completed=2 failed=1 incomplete=? complete=?"
    )


def test_ensure_docker_available_calls_docker_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_call_docker(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(["docker", *args], 0, stdout="16")

    monkeypatch.setattr(demo_requirements, "call_docker", fake_call_docker)

    demo_requirements.ensure_docker_available(reason="needed for postgres")

    assert calls == [("version", "--format", "{{.Server.Version}}")]


def test_ensure_docker_available_exits_with_demo_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_call_docker(*args: str) -> subprocess.CompletedProcess[str]:
        _ = args
        raise DockerUnavailableError()

    monkeypatch.setattr(demo_requirements, "call_docker", fake_call_docker)

    with pytest.raises(typer.Exit) as exc_info:
        demo_requirements.ensure_docker_available(
            reason="needed for postgres",
            recovery_hint="start Docker",
        )

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Docker is required but unavailable." in output
    assert "needed for postgres" in output
    assert "start Docker" in output


def test_create_demo_project_replaces_existing_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "maybe_get_project",
        lambda name: ProjectInfo(name=name),
    )
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    def fake_create_project(request: CreateProjectRequest) -> ProjectInfo:
        return ProjectInfo(name=request.project_name, port=5500)

    monkeypatch.setattr(demo_projects, "create_project", fake_create_project)

    project = demo_projects.create_demo_project(
        "demo",
        replace_existing=True,
    )

    assert destroyed == ["demo"]
    assert project.name == "demo"


def test_create_demo_project_can_preserve_existing_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "maybe_get_project",
        lambda name: ProjectInfo(name=name),
    )
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    def fake_create_project(request: CreateProjectRequest) -> ProjectInfo:
        return ProjectInfo(name=request.project_name, port=5500)

    monkeypatch.setattr(demo_projects, "create_project", fake_create_project)

    demo_projects.create_demo_project("demo")

    assert destroyed == []


def test_require_demo_project_dsn_returns_dsn() -> None:
    project = ProjectInfo(name="demo", port=5500)

    assert demo_projects.require_demo_project_dsn(project) == (
        "postgresql://postgres:postgres@localhost:5500/dr_llm"
    )


def test_require_demo_project_dsn_raises_for_missing_dsn() -> None:
    project = ProjectInfo(name="demo")

    with pytest.raises(RuntimeError, match="Demo project 'demo' has no DSN"):
        demo_projects.require_demo_project_dsn(project)


def test_prepare_demo_dsn_uses_existing_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_docker_check(
        *, reason: str, recovery_hint: str | None = None
    ) -> None:
        _ = (reason, recovery_hint)
        raise AssertionError("Docker should not be checked")

    monkeypatch.setattr(
        demo_projects,
        "ensure_docker_available",
        fail_docker_check,
    )

    lease = demo_projects.prepare_demo_dsn(
        dsn="postgresql://localhost/demo",
        project_prefix="demo",
        docker_reason="needed for postgres",
    )

    assert lease == demo_projects.DemoDsnLease(
        dsn="postgresql://localhost/demo"
    )


def test_prepare_demo_dsn_creates_disposable_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_checks: list[tuple[str, str | None]] = []
    created: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "ensure_docker_available",
        lambda *, reason, recovery_hint=None: docker_checks.append(
            (reason, recovery_hint)
        ),
    )
    monkeypatch.setattr(
        demo_projects,
        "temporary_demo_project_name",
        lambda prefix: f"{prefix}_abc123",
    )

    def fake_create_demo_project(project_name: str) -> ProjectInfo:
        created.append(project_name)
        return ProjectInfo(name=project_name, port=5500)

    monkeypatch.setattr(
        demo_projects,
        "create_demo_project",
        fake_create_demo_project,
    )

    lease = demo_projects.prepare_demo_dsn(
        dsn=None,
        project_prefix="demo",
        docker_reason="needed for postgres",
        docker_recovery_hint="start Docker",
    )

    assert docker_checks == [("needed for postgres", "start Docker")]
    assert created == ["demo_abc123"]
    assert lease == demo_projects.DemoDsnLease(
        dsn="postgresql://postgres:postgres@localhost:5500/dr_llm",
        project_name="demo_abc123",
        should_destroy_project=True,
    )


def test_prepare_demo_dsn_can_keep_named_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "ensure_docker_available",
        lambda *, reason, recovery_hint=None: None,
    )

    def fake_create_demo_project(project_name: str) -> ProjectInfo:
        created.append(project_name)
        return ProjectInfo(name=project_name, port=5500)

    monkeypatch.setattr(
        demo_projects,
        "create_demo_project",
        fake_create_demo_project,
    )

    lease = demo_projects.prepare_demo_dsn(
        dsn=None,
        project_prefix="demo",
        project_name="inspect_me",
        keep_project=True,
        docker_reason="needed for postgres",
    )

    assert created == ["inspect_me"]
    assert lease.project_name == "inspect_me"
    assert lease.should_destroy_project is False


def test_prepare_demo_dsn_cleans_up_disposable_project_without_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "ensure_docker_available",
        lambda *, reason, recovery_hint=None: None,
    )
    monkeypatch.setattr(
        demo_projects,
        "create_demo_project",
        lambda project_name: ProjectInfo(name=project_name),
    )
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    with pytest.raises(RuntimeError, match="Demo project 'broken' has no DSN"):
        demo_projects.prepare_demo_dsn(
            dsn=None,
            project_prefix="demo",
            project_name="broken",
            docker_reason="needed for postgres",
        )

    assert destroyed == ["broken"]


def test_cleanup_demo_dsn_destroys_only_owned_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    demo_projects.cleanup_demo_dsn(
        demo_projects.DemoDsnLease(
            dsn="postgresql://localhost/demo",
            project_name="temporary",
            should_destroy_project=True,
        )
    )
    demo_projects.cleanup_demo_dsn(
        demo_projects.DemoDsnLease(
            dsn="postgresql://localhost/demo",
            project_name="kept",
            should_destroy_project=False,
        )
    )
    demo_projects.cleanup_demo_dsn(
        demo_projects.DemoDsnLease(dsn="postgresql://localhost/demo")
    )

    assert destroyed == ["temporary"]


def test_temporary_demo_project_destroys_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "create_demo_project",
        lambda name: ProjectInfo(name=name, port=5500),
    )
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    with demo_projects.temporary_demo_project("demo") as project:
        assert project.name == "demo"

    assert destroyed == ["demo"]


def test_temporary_demo_project_destroys_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []

    monkeypatch.setattr(
        demo_projects,
        "create_demo_project",
        lambda name: ProjectInfo(name=name, port=5500),
    )
    monkeypatch.setattr(demo_projects, "destroy_project", destroyed.append)

    with pytest.raises(RuntimeError, match="boom"):
        with demo_projects.temporary_demo_project("demo"):
            raise RuntimeError("boom")

    assert destroyed == ["demo"]


def test_run_dr_llm_json_returns_parsed_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "cmd": cmd,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"ok": true}',
            stderr="",
        )

    monkeypatch.setattr(demo_cli_calls.subprocess, "run", fake_run)

    result = demo_cli_calls.run_dr_llm_json("models", "list", timeout=42)

    assert result == {"ok": True}
    assert calls == [
        {
            "cmd": ["uv", "run", "dr-llm", "models", "list"],
            "capture_output": True,
            "text": True,
            "timeout": 42,
        }
    ]


def test_run_dr_llm_json_raises_for_failed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        _ = capture_output, text, timeout
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="no catalog",
        )

    monkeypatch.setattr(demo_cli_calls.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="CLI command failed: models list"):
        demo_cli_calls.run_dr_llm_json("models", "list")


def test_run_dr_llm_streaming_uses_visible_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    commands: list[str] = []

    def fake_command(cmd: str) -> None:
        commands.append(cmd)

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        text: bool,
        stderr: int,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "cmd": cmd,
                "check": check,
                "text": text,
                "stderr": stderr,
            }
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(demo_cli_calls, "command", fake_command)
    monkeypatch.setattr(demo_cli_calls.subprocess, "run", fake_run)

    demo_cli_calls.run_dr_llm_streaming(
        "models", "sync", "--provider", "openai"
    )

    assert commands == ["uv run dr-llm models sync --provider openai"]
    assert calls == [
        {
            "cmd": [
                "uv",
                "run",
                "dr-llm",
                "models",
                "sync",
                "--provider",
                "openai",
            ],
            "check": True,
            "text": True,
            "stderr": subprocess.PIPE,
        }
    ]


def test_run_dr_llm_streaming_reports_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        text: bool,
        stderr: int,
    ) -> subprocess.CompletedProcess[str]:
        _ = check, text, stderr
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=cmd,
            stderr="bad provider",
        )

    monkeypatch.setattr(demo_cli_calls, "command", lambda cmd: None)
    monkeypatch.setattr(demo_cli_calls.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="exited with status 2: bad provider",
    ):
        demo_cli_calls.run_dr_llm_streaming(
            "models",
            "sync",
            "--provider",
            "missing",
        )


def test_model_json_helpers_build_expected_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_run_dr_llm_json(
        *args: str,
        timeout: int = demo_cli_calls.DEFAULT_CLI_TIMEOUT,
    ) -> dict[str, Any]:
        calls.append((args, timeout))
        if args[:2] == ("models", "list"):
            return {"models": [{"model": "gpt-demo"}]}
        return {"ok": True}

    monkeypatch.setattr(
        demo_cli_calls,
        "run_dr_llm_json",
        fake_run_dr_llm_json,
    )

    assert demo_cli_calls.sync_models_json("openai") == {"ok": True}
    assert demo_cli_calls.list_models_json("openai") == [{"model": "gpt-demo"}]
    assert demo_cli_calls.show_model_json("openai", "gpt-demo") == {"ok": True}

    assert calls == [
        (("models", "sync", "--provider", "openai", "--verbose"), 120),
        (("models", "list", "--provider", "openai", "--json"), 120),
        (
            (
                "models",
                "show",
                "--provider",
                "openai",
                "--model",
                "gpt-demo",
            ),
            120,
        ),
    ]


def test_query_json_appends_caller_supplied_extra_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_run_dr_llm_json(
        *args: str,
        timeout: int = demo_cli_calls.DEFAULT_CLI_TIMEOUT,
    ) -> dict[str, Any]:
        calls.append((args, timeout))
        return {"text": "ok"}

    monkeypatch.setattr(
        demo_cli_calls,
        "run_dr_llm_json",
        fake_run_dr_llm_json,
    )

    result = demo_cli_calls.query_json(
        "openai",
        "gpt-demo",
        "hello",
        timeout=300,
        extra_args=["--effort", "low"],
    )

    assert result == {"text": "ok"}
    assert calls == [
        (
            (
                "query",
                "--provider",
                "openai",
                "--model",
                "gpt-demo",
                "--message",
                "hello",
                "--effort",
                "low",
            ),
            300,
        )
    ]
