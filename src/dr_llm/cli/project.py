from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from dr_llm.project.backup import backup_project, restore_project
from dr_llm.project.docker import (
    create_project,
    destroy_project,
    get_project,
    list_projects,
    start_project,
    stop_project,
)

project_app = typer.Typer(help="Manage isolated dr-llm project databases")


@project_app.command("create")
def project_create(
    name: str = typer.Argument(
        ..., help="Project name (used in container/volume naming)"
    ),
) -> None:
    """Create a new project with its own Postgres container and persistent volume."""
    try:
        info = create_project(name)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(info.model_dump(mode="json", exclude_none=True), indent=2))


@project_app.command("list")
def project_list() -> None:
    """List all dr-llm projects."""
    projects = list_projects()
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
    """Start a stopped project's container."""
    try:
        info = start_project(name)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"Project '{name}' is running on port {info.port}", fg=typer.colors.GREEN
    )


@project_app.command("stop")
def project_stop(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Stop a project's container (data is preserved)."""
    try:
        stop_project(name)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Project '{name}' stopped. Data is preserved.", fg=typer.colors.GREEN)


@project_app.command("use")
def project_use(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Print the export command to set DR_LLM_DATABASE_URL for a project."""
    info = get_project(name)
    if info is None:
        typer.secho(f"Project '{name}' not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if info.status != "running":
        typer.secho(
            f"Project '{name}' is {info.status} — start it first",
            fg=typer.colors.YELLOW,
            err=True,
        )
    typer.echo(f"export DR_LLM_DATABASE_URL={info.dsn}")


@project_app.command("destroy")
def project_destroy(
    name: str = typer.Argument(..., help="Project name"),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Permanently destroy a project's container and all its data."""
    if not yes_really_delete_everything:
        typer.confirm(
            f"This will permanently delete ALL data for project '{name}'. Continue?",
            abort=True,
        )
    try:
        destroy_project(name)
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
    """Back up a project's database to a gzipped SQL file."""
    try:
        path = backup_project(name, output_dir=output_dir)
    except (RuntimeError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Backup saved to {path}", fg=typer.colors.GREEN)


@project_app.command("restore")
def project_restore(
    name: str = typer.Argument(..., help="Project name"),
    backup_file: Path = typer.Argument(
        ..., help="Path to backup file (.sql or .sql.gz)"
    ),
) -> None:
    """Restore a project's database from a backup file."""
    try:
        restore_project(name, backup_file)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Restored '{name}' from {backup_file}", fg=typer.colors.GREEN)
