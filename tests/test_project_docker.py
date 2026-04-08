from __future__ import annotations

import io
import os
import subprocess
import threading
from datetime import UTC, datetime
from typing import cast

import pytest

import dr_llm.project.docker_inspect as docker_inspect
import dr_llm.project.docker_lifecycle as docker_lifecycle
import dr_llm.project.docker_psql as docker_psql
import dr_llm.project.docker_runner as docker_runner
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
    DockerProjectMetadata,
)
from dr_llm.project.errors import DockerUnavailableError
from dr_llm.project.errors import (
    DockerCommandError,
    DockerContainerConflictError,
    DockerContainerNotFoundError,
    DockerContainerNotRunningError,
    DockerPortAllocatedError,
)


@pytest.mark.parametrize(
    ("stderr", "expected_type", "expected_message"),
    [
        (
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
            DockerUnavailableError,
            "Docker is not available. Install Docker or start the daemon.",
        ),
        (
            "Error response from daemon: No such container: demo",
            DockerContainerNotFoundError,
            "Docker container not found.",
        ),
        (
            "Error: No such object: demo",
            DockerContainerNotFoundError,
            "Docker container not found.",
        ),
        (
            "Error response from daemon: Container abc123 is not running",
            DockerContainerNotRunningError,
            "Docker container is not running.",
        ),
        (
            'Conflict. The container name "/demo" is already in use by container abc123.',
            DockerContainerConflictError,
            "Docker container name is already in use.",
        ),
        (
            "Bind for 0.0.0.0:5500 failed: port is already allocated",
            DockerPortAllocatedError,
            "Docker host port is already allocated.",
        ),
    ],
)
def test_docker_error_maps_common_failures(
    stderr: str,
    expected_type: type[Exception],
    expected_message: str,
) -> None:
    err = docker_runner.docker_error(("ps",), stderr)

    assert isinstance(err, expected_type)
    assert str(err) == expected_message


def test_call_docker_uses_text_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        args: list[str],
        *,
        input: bytes | None = None,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ["docker", "ps"]
        assert input is None
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is None
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)

    result = docker_runner.call_docker("ps")

    assert result.stdout == "ok"


def test_call_docker_bytes_uses_binary_mode_and_forwards_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        args: list[str],
        *,
        input: bytes | None = None,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        assert args == ["docker", "exec", "demo", "psql"]
        assert input == b"select 1;\n"
        assert capture_output is True
        assert text is False
        assert check is False
        assert env is None
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=b"ok",
            stderr=b"",
        )

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)

    result = docker_runner.call_docker_bytes(
        "exec",
        "demo",
        "psql",
        input=b"select 1;\n",
    )

    assert result.stdout == b"ok"


def test_call_docker_bytes_decodes_stderr_before_mapping_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        input: bytes | None = None,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = (input, capture_output, text, check, env)
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout=b"",
            stderr="bad-\u03b2".encode(),
        )

    def fake_docker_error(args: tuple[str, ...], stderr: str) -> DockerCommandError:
        observed["args"] = args
        observed["stderr"] = stderr
        return DockerCommandError("boom")

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_runner, "docker_error", fake_docker_error)

    with pytest.raises(DockerCommandError, match="boom"):
        docker_runner.call_docker_bytes("exec", "demo", "psql")

    assert observed == {
        "args": ("exec", "demo", "psql"),
        "stderr": "bad-\u03b2",
    }


def test_call_docker_start_is_idempotent_for_running_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == ("start", "demo")
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: container demo is already running",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    docker_lifecycle.call_docker_start("demo")


def test_call_docker_stop_is_idempotent_for_stopped_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == ("stop", "demo")
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: container demo is not running",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    docker_lifecycle.call_docker_stop("demo")


def test_wait_docker_ready_returns_immediately_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[int] = []

    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == (
            "exec",
            "demo",
            "pg_isready",
            "-U",
            "postgres",
            "-d",
            "dr_llm",
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout="accepting connections",
            stderr="",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)
    monkeypatch.setattr(
        docker_lifecycle, "sleep", lambda seconds: sleep_calls.append(seconds)
    )

    status = docker_lifecycle.wait_docker_ready("demo", "postgres", "dr_llm")

    assert status == ContainerStatus.RUNNING
    assert sleep_calls == []


def test_wait_docker_ready_raises_missing_container_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[int] = []

    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == (
            "exec",
            "demo",
            "pg_isready",
            "-U",
            "postgres",
            "-d",
            "dr_llm",
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: demo",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)
    monkeypatch.setattr(
        docker_lifecycle, "sleep", lambda seconds: sleep_calls.append(seconds)
    )

    with pytest.raises(DockerContainerNotFoundError):
        docker_lifecycle.wait_docker_ready("demo", "postgres", "dr_llm")

    assert sleep_calls == []


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def fetchone(self) -> tuple[int]:
        return (1,)

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_wait_dsn_ready_returns_immediately_on_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    connect_calls: list[tuple[str, int]] = []

    def fake_connect(dsn: str, *, connect_timeout: int = 0) -> _FakeConnection:
        connect_calls.append((dsn, connect_timeout))
        return _FakeConnection()

    monkeypatch.setattr(docker_lifecycle.psycopg, "connect", fake_connect)
    monkeypatch.setattr(
        docker_lifecycle, "sleep", lambda seconds: sleep_calls.append(seconds)
    )

    docker_lifecycle.wait_dsn_ready(
        "postgresql://postgres:postgres@localhost:5500/dr_llm"
    )

    assert connect_calls == [
        ("postgresql://postgres:postgres@localhost:5500/dr_llm", 2)
    ]
    assert sleep_calls == []


def test_wait_dsn_ready_retries_until_postgres_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three OperationalErrors followed by a successful connect should
    return cleanly without raising."""
    import psycopg

    sleep_calls: list[float] = []
    attempts = {"count": 0}

    def fake_connect(dsn: str, *, connect_timeout: int = 0) -> _FakeConnection:
        _ = (dsn, connect_timeout)
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise psycopg.OperationalError("server closed the connection")
        return _FakeConnection()

    monkeypatch.setattr(docker_lifecycle.psycopg, "connect", fake_connect)
    monkeypatch.setattr(
        docker_lifecycle, "sleep", lambda seconds: sleep_calls.append(seconds)
    )

    docker_lifecycle.wait_dsn_ready(
        "postgresql://postgres:postgres@localhost:5500/dr_llm",
        poll_interval_s=0.01,
    )

    assert attempts["count"] == 4
    # 3 sleeps (after each of the 3 failed attempts; success skips sleep).
    assert sleep_calls == [0.01, 0.01, 0.01]


def test_wait_dsn_ready_raises_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    monotonic_values = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    def fake_connect(dsn: str, *, connect_timeout: int = 0) -> _FakeConnection:
        _ = (dsn, connect_timeout)
        raise psycopg.OperationalError("server closed the connection")

    monkeypatch.setattr(docker_lifecycle.psycopg, "connect", fake_connect)
    monkeypatch.setattr(docker_lifecycle, "sleep", lambda _seconds: None)
    monkeypatch.setattr(docker_lifecycle, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(DockerCommandError, match="did not accept SQL connections"):
        docker_lifecycle.wait_dsn_ready(
            "postgresql://postgres:postgres@localhost:5500/dr_llm",
            timeout_seconds=2,
            poll_interval_s=0.01,
        )


def test_get_docker_project_metadata_raises_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == (
            "inspect",
            "--format",
            DockerProjectMetadata.inspect_format(),
            "demo",
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    with pytest.raises(DockerUnavailableError):
        docker_inspect.get_docker_project_metadata("demo")


def test_get_docker_project_metadata_returns_none_for_missing_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: demo",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    assert docker_inspect.get_docker_project_metadata("demo") is None


def test_get_all_docker_project_metadata_inspects_each_listed_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 4, 7, 12, 34, 56, tzinfo=UTC)
    inspect_calls: list[str] = []

    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        if args[0] == "ps":
            assert args == (
                "ps",
                "-a",
                "--filter",
                f"label={DockerProjectMetadata.name_label_key}",
                "--format",
                "{{.Names}}",
            )
            return subprocess.CompletedProcess(
                args=["docker", *args],
                returncode=0,
                stdout="dr-llm-pg-demo\ndr-llm-pg-other\n",
                stderr="",
            )
        assert args[:3] == (
            "inspect",
            "--format",
            DockerProjectMetadata.inspect_format(),
        )
        container = args[3]
        inspect_calls.append(container)
        # Label values include "," and "=" to prove the old comma-split parser
        # would have corrupted them — JSON inspect output handles them safely.
        labels = (
            f'{{"dr-llm.project.name":"{container}",'
            '"dr-llm.project.port":"5500",'
            f'"dr-llm.project.created-at":"{created_at.isoformat()}",'
            '"unrelated":"a,b=c"}'
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout=f'[{labels},"running"]',
            stderr="",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    metadata = docker_inspect.get_all_docker_project_metadata()

    assert inspect_calls == ["dr-llm-pg-demo", "dr-llm-pg-other"]
    assert [m.name for m in metadata] == ["dr-llm-pg-demo", "dr-llm-pg-other"]
    assert all(m.port == 5500 for m in metadata)
    assert all(m.created_at == created_at for m in metadata)
    assert all(m.status == ContainerStatus.RUNNING for m in metadata)


def test_get_all_docker_project_metadata_skips_containers_removed_during_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        if args[0] == "ps":
            return subprocess.CompletedProcess(
                args=["docker", *args],
                returncode=0,
                stdout="dr-llm-pg-gone\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: dr-llm-pg-gone",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    assert docker_inspect.get_all_docker_project_metadata() == []


def test_get_docker_project_metadata_parses_datetime_created_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 4, 7, 12, 34, 56, tzinfo=UTC)

    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == (
            "inspect",
            "--format",
            DockerProjectMetadata.inspect_format(),
            "demo",
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout=(
                '[{"dr-llm.project.name":"demo","dr-llm.project.port":"5500",'
                f'"dr-llm.project.created-at":"{created_at.isoformat()}"}},"running"]'
            ),
            stderr="",
        )

    monkeypatch.setattr(docker_runner, "call_docker", fake_call_docker)

    metadata = docker_inspect.get_docker_project_metadata("demo")

    assert metadata is not None
    assert metadata.name == "demo"
    assert metadata.port == 5500
    assert metadata.created_at == created_at
    assert metadata.status == ContainerStatus.RUNNING


def test_docker_project_create_metadata_builds_expected_run_args() -> None:
    created_at = datetime(2026, 4, 7, 12, 34, 56, tzinfo=UTC)
    project = DockerProjectCreateMetadata(
        name="demo",
        port=5500,
        created_at=created_at,
    )

    assert project.docker_run_args() == [
        "-p",
        "5500:5432",
        "--label",
        "dr-llm.project.name=demo",
        "--label",
        "dr-llm.project.port=5500",
        "--label",
        f"dr-llm.project.created-at={created_at.isoformat()}",
    ]


def test_docker_project_metadata_round_trips_create_labels() -> None:
    created_at = datetime(2026, 4, 7, 12, 34, 56, tzinfo=UTC)
    project = DockerProjectCreateMetadata(
        name="demo",
        port=5500,
        created_at=created_at,
    )

    metadata = DockerProjectMetadata.from_labels_status(
        labels=project.to_labels(),
        status="running",
    )

    assert metadata.name == "demo"
    assert metadata.port == 5500
    assert metadata.created_at == created_at
    assert metadata.status == ContainerStatus.RUNNING


def test_create_project_container_passes_password_via_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    created_at = datetime(2026, 4, 7, 12, 34, 56, tzinfo=UTC)
    monkeypatch.setenv("POSTGRES_PASSWORD", "outer")
    monkeypatch.setenv("UNRELATED_VAR", "from-parent")

    def fake_call_docker(
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        observed["check"] = check
        observed["env"] = env
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout="created",
            stderr="",
        )

    monkeypatch.setattr(docker_lifecycle, "call_docker", fake_call_docker)

    docker_lifecycle.create_project_container(
        volume_name="demo-volume",
        container_name="demo-container",
        db_name="dr_llm",
        db_user="postgres",
        db_password="inner-secret",
        docker_image="postgres:16",
        project=DockerProjectCreateMetadata(
            name="demo",
            port=5500,
            created_at=created_at,
        ),
    )

    assert observed["args"] == (
        "run",
        "-d",
        "--name",
        "demo-container",
        "-v",
        "demo-volume:/var/lib/postgresql/data",
        "-e",
        "POSTGRES_DB=dr_llm",
        "-e",
        "POSTGRES_USER=postgres",
        "-e",
        "POSTGRES_PASSWORD",
        "-p",
        "5500:5432",
        "--label",
        "dr-llm.project.name=demo",
        "--label",
        "dr-llm.project.port=5500",
        "--label",
        f"dr-llm.project.created-at={created_at.isoformat()}",
        "postgres:16",
    )
    assert observed["check"] is True
    env = cast(dict[str, str], observed["env"])
    assert env["POSTGRES_PASSWORD"] == "inner-secret"
    assert env["UNRELATED_VAR"] == "from-parent"
    assert os.environ["POSTGRES_PASSWORD"] == "outer"


def test_call_docker_forwards_env_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        input: bytes | None = None,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        observed["env"] = env
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)

    docker_runner.call_docker("ps", env={"FOO": "bar"})

    assert observed == {"args": ["docker", "ps"], "env": {"FOO": "bar"}}


def test_call_docker_destroy_attempts_volume_cleanup_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def fake_call_docker(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, check))
        if args[:2] == ("rm", "-f"):
            return subprocess.CompletedProcess(
                args=["docker", *args],
                returncode=1,
                stdout="",
                stderr="Error response from daemon: No such container: demo",
            )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout="demo-volume",
            stderr="",
        )

    monkeypatch.setattr(docker_lifecycle, "call_docker", fake_call_docker)

    with pytest.raises(DockerContainerNotFoundError):
        docker_lifecycle.call_docker_destroy("demo", "demo-volume")

    assert calls == [
        (("rm", "-f", "demo"), False),
        (("volume", "rm", "demo-volume"), False),
    ]


def test_docker_swap_in_db_drops_temp_db_when_stream_restore_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_psql_admin(
        container_name: str,
        db_user: str,
        *sql_commands: str,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = (container_name, db_user)
        calls.append(("admin", sql_commands))
        return subprocess.CompletedProcess(
            args=["docker", "exec", container_name, "psql"],
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(docker_psql, "_call_docker_psql_admin", fake_psql_admin)

    def fake_psql_input_stream(
        container_name: str,
        db_user: str,
        db_name: str,
        sql_stream: io.BytesIO,
    ) -> None:
        _ = (container_name, db_user)
        assert sql_stream.read() == b"select 1;\n"
        calls.append(("restore", (db_name,)))
        raise RuntimeError("restore failed")

    monkeypatch.setattr(
        docker_psql,
        "call_docker_psql_input_stream",
        fake_psql_input_stream,
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        docker_psql.docker_swap_in_db(
            sql_stream=io.BytesIO(b"select 1;\n"),
            container_name="demo",
            db_user="postgres",
            target_db_name="dr_llm",
        )

    assert len(calls) == 3
    assert calls[0] == ("admin", (f'CREATE DATABASE "{calls[1][1][0]}";',))
    assert calls[1] == ("restore", (calls[1][1][0],))
    assert calls[2] == ("admin", (f'DROP DATABASE IF EXISTS "{calls[1][1][0]}";',))


def test_docker_swap_in_db_creates_restores_and_swaps_temp_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_psql_admin(
        container_name: str,
        db_user: str,
        *sql_commands: str,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = (container_name, db_user)
        calls.append(("admin", sql_commands))
        return subprocess.CompletedProcess(
            args=["docker", "exec", container_name, "psql"],
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    def fake_psql_input_stream(
        container_name: str,
        db_user: str,
        db_name: str,
        sql_stream: io.BytesIO,
    ) -> None:
        _ = (container_name, db_user)
        assert sql_stream.read() == b"select 1;\n"
        calls.append(("restore", (db_name,)))

    monkeypatch.setattr(docker_psql, "_call_docker_psql_admin", fake_psql_admin)
    monkeypatch.setattr(
        docker_psql,
        "call_docker_psql_input_stream",
        fake_psql_input_stream,
    )

    docker_psql.docker_swap_in_db(
        sql_stream=io.BytesIO(b"select 1;\n"),
        container_name="demo",
        db_user="postgres",
        target_db_name="dr_llm",
    )

    swap_in_db = calls[1][1][0]
    assert calls == [
        ("admin", (f'CREATE DATABASE "{swap_in_db}";',)),
        ("restore", (swap_in_db,)),
        (
            "admin",
            (
                'DROP DATABASE IF EXISTS "dr_llm";',
                f'ALTER DATABASE "{swap_in_db}" RENAME TO "dr_llm";',
            ),
        ),
    ]


def test_call_docker_psql_input_stream_suppresses_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPipeWriter:
        def write(self, _data: object) -> int:
            raise BrokenPipeError

        def close(self) -> None:
            raise BrokenPipeError

    fake_process = subprocess.Popen[bytes].__new__(subprocess.Popen)
    fake_process.stdin = BrokenPipeWriter()

    def fake_run_docker_process(
        args: tuple[str, ...],
        *,
        stdin: int | None = None,
        stdout: int | None = None,
        operation,
    ) -> None:
        del args, stdin, stdout
        operation(fake_process)

    monkeypatch.setattr(docker_psql, "_run_docker_process", fake_run_docker_process)

    docker_psql.call_docker_psql_input_stream(
        container_name="demo",
        db_user="postgres",
        db_name="dr_llm",
        sql_stream=io.BytesIO(b"select 1;\n"),
    )


def test_run_docker_process_reads_stderr_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr_started = threading.Event()

    class FakeProcess:
        def __init__(self) -> None:
            self.stderr = io.BytesIO(b"boom")

        def wait(self) -> int:
            return 1

        def kill(self) -> None:
            return None

    fake_process = FakeProcess()

    def fake_popen(*args, **kwargs):
        del args, kwargs
        return fake_process

    def fake_read_process_stderr(process) -> str:
        assert process is fake_process
        stderr_started.set()
        return "boom"

    monkeypatch.setattr(docker_psql.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(docker_psql, "read_process_stderr", fake_read_process_stderr)

    with pytest.raises(DockerCommandError, match="boom"):
        docker_psql._run_docker_process(
            ("exec", "demo", "psql"),
            operation=lambda process: (
                stderr_started.wait(0.5)
                or (_ for _ in ()).throw(AssertionError("stderr reader did not start"))
            ),
        )
