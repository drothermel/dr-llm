from __future__ import annotations

import json

import typer
from pydantic import ValidationError

from dr_llm.cli.common import handle_cli_errors
from dr_llm.pool.admin.deletion import (
    DeletePoolRequest,
    DeletePoolsByTokenRequest,
    delete_pool,
    delete_pools_by_token,
)
from dr_llm.pool.admin.discovery import discover_pools
from dr_llm.pool.admin.inspection import inspect_pool_dsn
from dr_llm.pool.errors import PoolError
from dr_llm.project.errors import ProjectError

pool_app = typer.Typer(help="Manage dr-llm pools")


@pool_app.command("list-dsn")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_list_dsn(
    dsn: str = typer.Option(
        ...,
        "--dsn",
        envvar="DR_LLM_DATABASE_URL",
        help="Postgres DSN to inspect. Defaults to DR_LLM_DATABASE_URL.",
    ),
) -> None:
    """List pools from any dr-llm Postgres database."""
    typer.echo(json.dumps({"pools": discover_pools(dsn)}, indent=2))


@pool_app.command("inspect-dsn")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_inspect_dsn(
    pool_name: str = typer.Argument(..., help="Pool name"),
    dsn: str = typer.Option(
        ...,
        "--dsn",
        envvar="DR_LLM_DATABASE_URL",
        help="Postgres DSN to inspect. Defaults to DR_LLM_DATABASE_URL.",
    ),
) -> None:
    """Inspect a pool from any dr-llm Postgres database."""
    inspection = inspect_pool_dsn(dsn=dsn, pool_name=pool_name)
    typer.echo(
        json.dumps(
            inspection.model_dump(mode="json", exclude_none=True),
            indent=2,
        )
    )


@pool_app.command("destroy")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_destroy(
    project_name: str = typer.Argument(..., help="Project name"),
    pool_name: str = typer.Argument(..., help="Pool name"),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    if not yes_really_delete_everything:
        typer.confirm(
            "This will permanently delete the pool and all of its tables. Continue?",
            abort=True,
        )

    result = delete_pool(
        DeletePoolRequest(project_name=project_name, pool_name=pool_name)
    )
    typer.echo(
        json.dumps(
            result.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
            indent=2,
        )
    )
    if not result.success:
        raise typer.Exit(1)


@pool_app.command("destroy-testish")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_destroy_testish(
    project_name: str = typer.Argument(..., help="Project name"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview matching pools without deleting them.",
    ),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    if not dry_run and not yes_really_delete_everything:
        typer.confirm(
            "This will permanently delete pools whose underscore-delimited "
            "tokens include test, tst, smoke, or demo. Continue?",
            abort=True,
        )

    result = delete_pools_by_token(
        DeletePoolsByTokenRequest(
            project_name=project_name,
            match_tokens=["test", "tst", "smoke", "demo"],
            dry_run=dry_run,
        )
    )
    typer.echo(
        json.dumps(
            result.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
            indent=2,
        )
    )
    if not result.success:
        raise typer.Exit(1)
