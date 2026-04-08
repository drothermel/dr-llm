from __future__ import annotations

import io
import subprocess
from typing import IO, cast

import pytest

import dr_llm.project.docker as docker_module


@pytest.mark.parametrize(
    ("stderr", "expected_type", "expected_message"),
    [
        (
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
            RuntimeError,
            "Docker is not available. Install Docker or start the daemon.",
        ),
        (
            "Error response from daemon: No such container: demo",
            docker_module.DockerContainerNotFoundError,
            "Docker container not found.",
        ),
        (
            "Error response from daemon: Container abc123 is not running",
            docker_module.DockerContainerNotRunningError,
            "Docker container is not running.",
        ),
        (
            'Conflict. The container name "/demo" is already in use by container abc123.',
            docker_module.DockerContainerConflictError,
            "Docker container name is already in use.",
        ),
        (
            "Bind for 0.0.0.0:5500 failed: port is already allocated",
            docker_module.DockerPortAllocatedError,
            "Docker host port is already allocated.",
        ),
    ],
)
def test_docker_error_maps_common_failures(
    stderr: str,
    expected_type: type[RuntimeError],
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

    with pytest.raises(docker_module.DockerContainerNotFoundError):
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
