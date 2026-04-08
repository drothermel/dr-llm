from __future__ import annotations

import subprocess

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

