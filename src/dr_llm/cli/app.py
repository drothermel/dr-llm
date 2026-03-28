from __future__ import annotations

import os

import typer

from dr_llm.project.docker import get_project

from .models import models_app
from .project import project_app
from .providers import register as register_providers
from .query import register as register_query
from .run import run_app

app = typer.Typer()

app.add_typer(run_app, name="run")
app.add_typer(models_app, name="models")
app.add_typer(project_app, name="project")
register_providers(app)
register_query(app)


@app.callback()
def main(
    project: str | None = typer.Option(None, help="Use a named project's database."),
) -> None:
    """dr-llm CLI"""
    if project is not None:
        info = get_project(project)
        if info is None:
            typer.secho(f"Project '{project}' not found", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        if info.status != "running":
            typer.secho(
                f"Project '{project}' is {info.status} - start it first",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        os.environ["DR_LLM_DATABASE_URL"] = info.dsn
