from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from dr_llm.cli.common import handle_cli_errors
from dr_llm.project.errors import ProjectError
from dr_llm.project.project_info import ProjectInfo

project_app = typer.Typer(help="Manage isolated dr-llm project databases")


@project_app.command("create")
@handle_cli_errors(ProjectError)
def project_create(
    name: str = typer.Argument(
        ..., help="Project name (used in container/volume naming)"
    ),
) -> None:
    """Create a new project with its own Postgres container and persistent volume."""
    project_info = ProjectInfo.create_new(name)
    typer.echo(
        json.dumps(project_info.model_dump(mode="json", exclude_none=True), indent=2)
    )


@project_app.command("list")
@handle_cli_errors(ProjectError)
def project_list() -> None:
    """List all dr-llm projects."""
    projects = ProjectInfo.list_all()
    if not projects:
        typer.echo("No projects found.")
        return
    port_values = [
        str(project.port) if project.port is not None else "-" for project in projects
    ]
    name_w = max(len(project.name) for project in projects)
    port_w = max(len(port) for port in port_values)
    header = f"{'NAME':<{name_w}}  {'PORT':<{port_w}}  STATUS"
    typer.echo(header)
    typer.echo("-" * len(header))
    for project, port in zip(projects, port_values, strict=True):
        typer.echo(f"{project.name:<{name_w}}  {port:<{port_w}}  {project.status}")


@project_app.command("start")
@handle_cli_errors(ProjectError)
def project_start(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    project_info = ProjectInfo.start(name)
    typer.secho(
        f"Project '{name}' is running on port {project_info.port}",
        fg=typer.colors.GREEN,
    )


@project_app.command("stop")
@handle_cli_errors(ProjectError)
def project_stop(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    ProjectInfo.stop(name)
    typer.secho(f"Project '{name}' stopped. Data is preserved.", fg=typer.colors.GREEN)


@project_app.command("use")
@handle_cli_errors(ProjectError)
def project_use(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Print the export command to set DR_LLM_DATABASE_URL for a project."""
    project_info = ProjectInfo.get_by_name(name)
    if not project_info.running:
        typer.secho(
            f"Project '{name}' is {project_info.status} - start it first",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"export DR_LLM_DATABASE_URL={project_info.dsn}")


@project_app.command("destroy")
@handle_cli_errors(ProjectError)
def project_destroy(
    name: str = typer.Argument(..., help="Project name"),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    if not yes_really_delete_everything:
        typer.confirm(
            f"This will permanently delete ALL data for project '{name}'. Continue?",
            abort=True,
        )

    ProjectInfo.destroy(name)
    typer.secho(
        f"Project '{name}' destroyed (container + volume removed).",
        fg=typer.colors.RED,
    )


@project_app.command("backup")
@handle_cli_errors(ProjectError, FileNotFoundError)
def project_backup(
    name: str = typer.Argument(..., help="Project name"),
    output_dir: Path | None = typer.Option(None, help="Custom backup directory."),
) -> None:
    path = ProjectInfo.backup(name, output_dir)
    typer.secho(f"Backup saved to {path}", fg=typer.colors.GREEN)


@project_app.command("restore")
@handle_cli_errors(ProjectError, FileNotFoundError, subprocess.CalledProcessError)
def project_restore(
    name: str = typer.Argument(..., help="Project name"),
    backup_file: Path = typer.Argument(..., help="Path to backup file (.sql.gz)"),
) -> None:
    ProjectInfo.restore(name, backup_file)
    typer.secho(f"Restored '{name}' from {backup_file}", fg=typer.colors.GREEN)
