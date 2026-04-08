from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from dr_llm.project.project_info import ProjectInfo

project_app = typer.Typer(help="Manage isolated dr-llm project databases")


@project_app.command("create")
def project_create(
    name: str = typer.Argument(
        ..., help="Project name (used in container/volume naming)"
    ),
) -> None:
    """Create a new project with its own Postgres container and persistent volume."""
    try:
        project_info = ProjectInfo.create_new(name)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(project_info.model_dump(mode="json", exclude_none=True), indent=2)
    )


@project_app.command("list")
def project_list() -> None:
    """List all dr-llm projects."""
    projects = ProjectInfo.list_all()
    if not projects:
        typer.echo("No projects found.")
        return
    name_w = max(len(project.name) for project in projects)
    port_w = max(len(str(project.port)) for project in projects)
    header = f"{'NAME':<{name_w}}  {'PORT':<{port_w}}  STATUS"
    typer.echo(header)
    typer.echo("-" * len(header))
    for project in projects:
        typer.echo(
            f"{project.name:<{name_w}}  {project.port:<{port_w}}  {project.status}"
        )


@project_app.command("start")
def project_start(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    try:
        project_info = ProjectInfo(name=name)
        project_info.start()
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"Project '{name}' is running.",
        fg=typer.colors.GREEN,
    )


@project_app.command("stop")
def project_stop(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    try:
        project_info = ProjectInfo(name=name)
        project_info.stop()
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Project '{name}' stopped. Data is preserved.", fg=typer.colors.GREEN)


@project_app.command("use")
def project_use(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Print the export command to set DR_LLM_DATABASE_URL for a project."""
    try:
        project_info = ProjectInfo.get_by_name(name)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    if not project_info.running:
        typer.secho(
            f"Project '{name}' is {project_info.status} - start it first",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"export DR_LLM_DATABASE_URL={project_info.dsn}")


@project_app.command("destroy")
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

    try:
        project_info = ProjectInfo(name=name)
        project_info.destroy()
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"Project '{name}' destroyed (container + volume removed).",
        fg=typer.colors.RED,
    )


@project_app.command("backup")
def project_backup(
    name: str = typer.Argument(..., help="Project name"),
    output_dir: Path | None = typer.Option(None, help="Custom backup directory."),
) -> None:
    try:
        project_info = ProjectInfo(name=name)
        path = project_info.backup(output_dir)
    except (RuntimeError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Backup saved to {path}", fg=typer.colors.GREEN)


@project_app.command("restore")
def project_restore(
    name: str = typer.Argument(..., help="Project name"),
    backup_file: Path = typer.Argument(..., help="Path to backup file (.sql.gz)"),
) -> None:
    try:
        project_info = ProjectInfo(name=name)
        project_info.restore(backup_file)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Restored '{name}' from {backup_file}", fg=typer.colors.GREEN)
