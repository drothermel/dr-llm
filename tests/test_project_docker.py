from __future__ import annotations

import io
import os
import subprocess
from typing import IO, cast

import pytest

import dr_llm.project.docker as docker_module
from dr_llm.project.docker_project_metadata import (
    ContainerStatus,
    DockerProjectCreateMetadata,
)
from dr_llm.project.errors import DockerUnavailableError
from dr_llm.project.errors import (
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
    err = docker_module._docker_error(("ps",), stderr)

    assert isinstance(err, expected_type)
    assert str(err) == expected_message


def test_call_docker_start_is_idempotent_for_running_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == ("start", "demo")
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: container demo is already running",
        )

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    docker_module.call_docker_start("demo")


def test_call_docker_stop_is_idempotent_for_stopped_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == ("stop", "demo")
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: container demo is not running",
        )

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    docker_module.call_docker_stop("demo")


def test_wait_docker_ready_returns_immediately_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[int] = []

    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)
    monkeypatch.setattr(docker_module, "sleep", lambda seconds: sleep_calls.append(seconds))

    status = docker_module.wait_docker_ready("demo", "postgres", "dr_llm")

    assert status == ContainerStatus.RUNNING
    assert sleep_calls == []


def test_wait_docker_ready_raises_missing_container_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[int] = []

    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)
    monkeypatch.setattr(docker_module, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(DockerContainerNotFoundError):
        docker_module.wait_docker_ready("demo", "postgres", "dr_llm")

    assert sleep_calls == []


def test_get_docker_project_metadata_raises_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        _ = check
        assert args == (
            "inspect",
            "--format",
            "{{json .Config.Labels}}||{{json .State.Status}}",
            "demo",
        )
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
        )

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    with pytest.raises(DockerUnavailableError):
        docker_module.get_docker_project_metadata("demo", label_prefix="dr-llm.project")


def test_get_docker_project_metadata_returns_none_for_missing_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        _ = check
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: demo",
        )

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    assert (
        docker_module.get_docker_project_metadata("demo", label_prefix="dr-llm.project")
        is None
    )


def test_docker_project_create_metadata_builds_expected_run_args() -> None:
    project = DockerProjectCreateMetadata(
        label_prefix="dr-llm.project",
        name="demo",
        port=5500,
        created_at="2026-04-07T12:34:56+00:00",
    )

    assert project.docker_run_args() == [
        "-p",
        "5500:5432",
        "--label",
        "dr-llm.project.name=demo",
        "--label",
        "dr-llm.project.port=5500",
        "--label",
        "dr-llm.project.created-at=2026-04-07T12:34:56+00:00",
    ]


def test_temp_environ_sets_and_restores_missing_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with docker_module._temp_environ(POSTGRES_PASSWORD="secret"):
        assert os.environ["POSTGRES_PASSWORD"] == "secret"

    assert "POSTGRES_PASSWORD" not in os.environ


def test_temp_environ_restores_previous_env_var_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "original")

    with pytest.raises(RuntimeError, match="boom"):
        with docker_module._temp_environ(POSTGRES_PASSWORD="secret"):
            assert os.environ["POSTGRES_PASSWORD"] == "secret"
            raise RuntimeError("boom")

    assert os.environ["POSTGRES_PASSWORD"] == "original"


def test_create_project_container_uses_project_metadata_and_restores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("POSTGRES_PASSWORD", "outer")

    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        observed["check"] = check
        observed["password"] = os.environ["POSTGRES_PASSWORD"]
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=0,
            stdout="created",
            stderr="",
        )

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    docker_module.create_project_container(
        volume_name="demo-volume",
        container_name="demo-container",
        db_name="dr_llm",
        db_user="postgres",
        db_password="inner-secret",
        docker_image="postgres:16",
        project=DockerProjectCreateMetadata(
            label_prefix="dr-llm.project",
            name="demo",
            port=5500,
            created_at="2026-04-07T12:34:56+00:00",
        ),
    )

    assert observed == {
        "args": (
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
            "dr-llm.project.created-at=2026-04-07T12:34:56+00:00",
            "postgres:16",
        ),
        "check": True,
        "password": "inner-secret",
    }
    assert os.environ["POSTGRES_PASSWORD"] == "outer"


def test_call_docker_destroy_attempts_volume_cleanup_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def fake_call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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

    monkeypatch.setattr(docker_module, "call_docker", fake_call_docker)

    with pytest.raises(DockerContainerNotFoundError):
        docker_module.call_docker_destroy("demo", "demo-volume")

    assert calls == [
        (("rm", "-f", "demo"), False),
        (("volume", "rm", "demo-volume"), False),
    ]


def test_docker_swap_in_db_drops_temp_db_when_stream_restore_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        docker_module,
        "call_docker_psql_create_db",
        lambda container_name, db_user, db_name: calls.append(("create", db_name)),
    )

    def fake_psql_input_stream(
        container_name: str,
        db_user: str,
        db_name: str,
        sql_stream: io.BytesIO,
    ) -> None:
        _ = (container_name, db_user)
        assert sql_stream.read() == b"select 1;\n"
        calls.append(("restore", db_name))
        raise RuntimeError("restore failed")

    monkeypatch.setattr(
        docker_module,
        "call_docker_psql_input_stream",
        fake_psql_input_stream,
    )
    monkeypatch.setattr(
        docker_module,
        "call_docker_psql_swap_in_db",
        lambda container_name, db_user, target_db_name, swap_in_db: calls.append(
            ("swap", swap_in_db)
        ),
    )
    monkeypatch.setattr(
        docker_module,
        "call_docker_psql_drop_db",
        lambda container_name, db_user, db_name: calls.append(("drop", db_name)),
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        docker_module.docker_swap_in_db(
            sql_stream=cast(IO[bytes], io.BytesIO(b"select 1;\n")),
            container_name="demo",
            db_user="postgres",
            target_db_name="dr_llm",
        )

    assert len(calls) == 3
    assert calls[0][0] == "create"
    assert calls[1] == ("restore", calls[0][1])
    assert calls[2] == ("drop", calls[0][1])
