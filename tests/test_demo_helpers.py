from __future__ import annotations

import subprocess

import pytest
import typer

import dr_llm.demo.projects as demo_projects
import dr_llm.demo.requirements as demo_requirements
from dr_llm.project import CreateProjectRequest, ProjectInfo
from dr_llm.project.errors import DockerUnavailableError


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
